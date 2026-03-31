import argparse
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops

from model import Generator
from src.dataset import CHAR_TO_CLASS, NUM_CLASSES

# ── Digraph-aware tokenizer ───────────────────────────────────────────────────
DIGRAPHS = {'ու': 63, 'Ու': 24, 'Եվ': 36, 'և': 75}


def tokenize(word: str) -> list[tuple[str, int]]:
    tokens = []
    i = 0
    while i < len(word):
        # Ճանաչում ենք  Space-ը բառերն անջատելու համար
        if word[i] == ' ':
            tokens.append((' ', -1))  # -1 ինդեքսը ցույց է տալիս դատարկ տարածություն
            i += 1
            continue

        # Digraph ստուգում — 2 նիշ որ 1 տառ են կազմում (ու, Ու, Եվ, և)
        # Օրինակ՝ 'ուրախ' → [('ու',63), ('ր',71), ('ա',39), ('խ',51), ('հ',54)]
        # առանց դիգրաֆի կստանայինք ('ո' + 'ւ') — սխալ կլիներ
        two = word[i:i + 2]
        if two in DIGRAPHS:
            tokens.append((two, DIGRAPHS[two]))
            i += 2
            continue

        ch = word[i]
        if ch in CHAR_TO_CLASS:
            tokens.append((ch, CHAR_TO_CLASS[ch]))
        else:
            print(f"[WARNING] Character '{ch}' (U+{ord(ch):04X}) not in dataset, skipping.")
        i += 1
    return tokens


# ── Generation ────────────────────────────────────────────────────────────────

def generate_word(
        text: str,
        checkpoint_path: str,
        output_path: str = 'output.png',
        latent_dim: int = 100,
        img_size: int = 64,
        device: str = 'auto',
        pad: int = -18,  # Տառերի արանքի միացումը
        space_width: int = 30  # Բառերի արանքի հեռավորությունը (պրոբելի լայնությունը)
):
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    # 1. model-ի բեռնում
    G = Generator(num_classes=NUM_CLASSES, latent_dim=latent_dim, embed_dim=latent_dim).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev)  #.pt ֆայլը բեռնում ենք
    G.load_state_dict(ckpt['G_state_dict'])               # weight-երը տեղադրում ենք
    G.eval()                                              # Dropout/BatchNorm-ը անջատում ենք — inference mode

    # 2. Tokenize text
    tokens = tokenize(text)
    if not tokens:
        raise ValueError(f"No valid Armenian characters found in: '{text}'")

    # 3. Ֆիքսում ենք Z վեկտորը (Style Consistency)
    # Ամբողջ նախադասությունը կգեներացվի նույն աղմուկի վեկտորով, որպեսզի
    # ձեռագիրը (հաստությունը, թեքությունը) նույնը մնա բոլոր տառերի համար:
    fixed_noise = torch.randn(1, latent_dim, device=dev)

    # 4. Յուրաքանչյուր տառի գեներացիա
    elements = []
    with torch.no_grad():    # gradient չենք հաշվում — ավելի արագ, memory պահում
        for char_str, class_id in tokens:
            if class_id == -1:
                elements.append('SPACE')   # բացատ — նկար չի գեներացվում
            else:
                label = torch.tensor([class_id], device=dev)
                img_tensor = G(fixed_noise, label)   # Generator-ը գեներացնում է
                img_tensor = (img_tensor + 1) / 2.0  # [-1,1] → [0,1]
                pil_img = TF.to_pil_image(img_tensor.squeeze(0))  # tensor → PIL
                elements.append(pil_img)
    # elements = [<PIL 'Մ'>, <PIL 'ա'>, 'SPACE', <PIL 'բ'>, ...]

    # 5. Հաշվում ենք նկարի իրական ընդհանուր լայնությունը
    total_width = 0
    for i, el in enumerate(elements):
        if el == 'SPACE':
            total_width += space_width  # բացատ = 30px (default)
        else:
            total_width += el.width     # տառի լայնությունը (64px)
            # Բացասական pad (-18) կիրառում ենք ՄԻԱՅՆ այն դեպքում,
            # երբ հաջորդ տարրը նույնպես տառ է (որպեսզի կպնեն իրար)
            # տառ → տառ  : pad կիրառվում է  (կպնում են)
            # տառ → բացատ: pad ՉԻ կիրառվում (բացատը ինքն է բաժանում)
            if i < len(elements) - 1 and elements[i + 1] != 'SPACE':
                total_width += pad  # pad=-18 → 18px overlap

    total_width = max(img_size, total_width)  # նվազագույնը 64px

    # 6. Հավաքում ենք նկարը (Image Stitching)
    word_img = Image.new('L', (total_width, img_size), color=0)  # Start with black

    x_offset = 0  # ընթացիկ X դիրք
    for i, el in enumerate(elements):
        if el == 'SPACE':
            x_offset += space_width  # պարզապես առաջ ենք մղվում
        else:
            # Ժամանակավոր canvas — նույն չափի, ամբողջը սև
            temp_canvas = Image.new('L', (total_width, img_size), color=0)
            temp_canvas.paste(el, (x_offset, 0))  # տառը տեղադրում ենք ճիշտ X-ով
            # Use ImageChops.lighter to blend overlapping strokes correctly
            # lighter() — պիքսել առ պիքսել վերցնում է ավելի ՎԱՌԸ
            # Սև ֆոն (0) vs տառի պիքսել (200) → 200 հաղթում է
            # Overlap գոտում՝ երկու տառի վառ պիքսելները պահվում են
            word_img = ImageChops.lighter(word_img, temp_canvas)

            x_offset += el.width
            # Առաջ ենք գնում (տառերն իրար վրայով անցկացնում ենք) pad-ի չափով
            if i < len(elements) - 1 and elements[i + 1] != 'SPACE':
                x_offset += pad  # -18 → հաջորդ տառը 18px հետ է գալու

    # ── CV Processing for Quality ──
    arr = np.array(word_img).astype(np.uint8)

    # 1. Soften the GAN output to fill internal holes
    # Blur — GAN-ի "ծակերը" լրացնում, եզրերը մեղմացնում
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)

    # 2. Global threshold: Lower than before (40) to keep faint "erased" parts
    # 40 ցածր է (կարող էր 127 լինել), որպեսզի բաց գծերը չկորեն
    _, mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    # 3. Dilation: Physically thicken the strokes to ensure connections
    # 2×2 kernel, 1 iteration → մոտ 1-2px թանձրացում
    # Կպցնում է ընդհատված գծերը
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 4. Final smoothing for clean ink edges
    # Կրկին blur + threshold — dilate-ից առաջացած կոպիտ եզրերը հարթեցնում
    final_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    # 5. Create RGBA (Black Ink, Transparent BG)
    # Background: Alpha=0. Text: Alpha=255.
    # Ինչու RGBA և ոչ L (grayscale)?
    # → PNG-ն կարելի է ցանկացած ֆոնի վրա դնել (Word, Photoshop, web)
    rgba = np.zeros((*final_mask.shape, 4), dtype=np.uint8)
    rgba[..., 0:3] = 0  # R, G, B are all 0 for Black
    rgba[..., 3] = final_mask  # Alpha channel uses the mask itself

    result = Image.fromarray(rgba, 'RGBA')
    result.save(output_path)
    print(f"[Generate] Image saved to {output_path} with accurate word spacing and style consistency.")


def parse_args():
    p = argparse.ArgumentParser(description='Generate Armenian handwritten word image')
    p.add_argument('--text', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output', type=str, default='output.png')
    p.add_argument('--latent_dim', type=int, default=100)
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--pad', type=int, default=-18, help='Use -15 to -22 for cursive letters')
    p.add_argument('--space_width', type=int, default=30, help='Width of the space between words in pixels')
    p.add_argument('--device', type=str, default='auto')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_word(
        text=args.text,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        device=args.device,
        pad=args.pad,
        space_width=args.space_width
    )