"""
app.py — Հայերեն Ձեռագրի Գեներացիա  ·  Photoshop-style editor  (v6.5 FINAL UI)
═══════════════════════════════════════════════════════════════════
Changes:
  • Added explicit "Prev/Next Layer" buttons to fix UX confusion where users 
    were modifying Z-index instead of changing layer selection.
  • Renamed Z-index buttons to "🔼 Բերել առաջ" and "🔽 Տանել հետ".
  • Fixed Up/Down Pan arrows to standard logic.
  • Cleaned up debug logs.
"""

import torch
import numpy as np
import cv2
import gradio as gr
import json
import base64
import io
import re
from PIL import Image, ImageChops, ImageDraw, ImageColor, ImageOps
from model import Generator
from dataset import CHAR_TO_CLASS, NUM_CLASSES

# ── HEIC/HEIF support ─────────────────────────────────────────────────────────
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ HEIF/HEIC support enabled")
except ImportError:
    print("⚠️ pillow-heif not installed — HEIC files won't be supported")

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"checkpoints/checkpoint_epoch_0140.pt"
LATENT_DIM      = 100
DIGRAPHS        = {'ու': 63, 'Ու': 24, 'Եվ': 36, 'և': 75}
GALLERY_SEEDS      = [0, 7, 555, 42989, 256, 512, 999, 1337, 2024, 3333, 5050, 7777, 8888, 9999]
GALLERY_INIT_COUNT = 4
PREVIEW_TEXT    = "Բարև"
CHAR_NOISE_VARIATION = 0.0  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Init] Device: {device}")


def _parse_epoch(path: str) -> int:
    m = re.search(r'epoch[_\-]?(\d+)', path, re.IGNORECASE)
    return int(m.group(1)) if m else -1


try:
    G = Generator(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, embed_dim=LATENT_DIM).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()
    _epoch = _parse_epoch(CHECKPOINT_PATH)
    if _epoch > 200:
        print(f"⚠️  WARNING: checkpoint epoch {_epoch} > 200")
    else:
        print(f"✅ Model loaded  (epoch {_epoch})")
except Exception as e:
    print(f"❌ Model error: {e}")
    G = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def tokenize(word: str) -> list:
    tokens, i = [], 0
    while i < len(word):
        if word[i] == ' ':
            tokens.append((' ', -1)); i += 1; continue
        two = word[i:i+2]
        if two in DIGRAPHS:
            tokens.append((two, DIGRAPHS[two])); i += 2; continue
        ch = word[i]
        if ch in CHAR_TO_CLASS:
            tokens.append((ch, CHAR_TO_CLASS[ch]))
        else:
            pass
        i += 1
    return tokens


def calculate_brightness(image_region: Image.Image) -> float:
    arr = np.array(image_region.convert("RGB"))
    if arr.size == 0: return 255.0
    luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.mean(luminance))


def recolor(img_rgba: Image.Image, color_val: str) -> Image.Image:
    _, _, _, alpha = img_rgba.split()
    try:
        color_val = str(color_val).strip()
        if not color_val: color_val = "#ffffff"
        if color_val.startswith("rgb"):
            match = re.search(r'rgba?\(([^)]+)\)', color_val)
            if match:
                parts = [float(x.strip()) for x in match.group(1).split(',')]
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                r, g, b = 255, 255, 255
        else:
            r, g, b = ImageColor.getrgb(color_val)[:3]
    except Exception:
        r, g, b = 255, 255, 255
    solid = Image.new("RGB", img_rgba.size, (r, g, b))
    return Image.merge("RGBA", (*solid.split(), alpha))


def apply_stroke_and_color(img_rgba: Image.Image, color_val: str, stroke_width: int, stroke_color: str):
    colored_text = recolor(img_rgba, color_val)
    if stroke_width <= 0:
        return colored_text, 0
        
    pad = stroke_width
    w, h = colored_text.size
    new_size = (w + pad * 2, h + pad * 2)
    
    padded_img = Image.new("RGBA", new_size, (0, 0, 0, 0))
    padded_img.paste(colored_text, (pad, pad))
    
    arr = np.array(padded_img)
    alpha = arr[..., 3]
    
    kernel_size = stroke_width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_alpha = cv2.dilate(alpha, kernel, iterations=1)
    
    try:
        stroke_color = str(stroke_color).strip()
        if not stroke_color: stroke_color = "#000000"
        if stroke_color.startswith("rgb"):
            match = re.search(r'rgba?\(([^)]+)\)', stroke_color)
            if match:
                parts = [float(x.strip()) for x in match.group(1).split(',')]
                sr, sg, sb = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                sr, sg, sb = 0, 0, 0
        else:
            sr, sg, sb = ImageColor.getrgb(stroke_color)[:3]
    except:
        sr, sg, sb = 0, 0, 0
        
    stroke_img = np.zeros((*dilated_alpha.shape, 4), dtype=np.uint8)
    stroke_img[..., 0] = sr
    stroke_img[..., 1] = sg
    stroke_img[..., 2] = sb
    stroke_img[..., 3] = dilated_alpha
    
    stroke_pil = Image.fromarray(stroke_img, "RGBA")
    stroke_pil.alpha_composite(padded_img)
    return stroke_pil, pad


def _make_alpha_mask(raw_arr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(raw_arr, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _measure_word_width(word, letter_size, pad, space_w):
    tokens = tokenize(word)
    glyphs_count = sum(1 for _, cid in tokens if cid != -1)
    spaces_count  = sum(1 for _, cid in tokens if cid == -1)
    w = glyphs_count * letter_size + max(glyphs_count - 1, 0) * pad + spaces_count * space_w
    return max(letter_size, w)


def _wrap_words(text, wrap_width, letter_size, pad, space_w):
    words = text.split(' ')
    lines = []
    current = []

    def _flush():
        if current:
            line = ' '.join(w for w in current if w)
            if line.strip():
                lines.append(line)
            current.clear()

    for word in words:
        if not word: continue
        word_w = _measure_word_width(word, letter_size, pad, space_w)
        if word_w > wrap_width:
            _flush()
            chunk = ''
            for ch in word:
                test = chunk + ch
                if _measure_word_width(test, letter_size, pad, space_w) > wrap_width and chunk:
                    lines.append(chunk)
                    chunk = ch
                else:
                    chunk = test
            if chunk:
                current.append(chunk)
            continue
        test_text = ' '.join(w for w in (current + [word]) if w)
        if current and _measure_word_width(test_text, letter_size, pad, space_w) > wrap_width:
            _flush()
            current.append(word)
        else:
            current.append(word)
    _flush()
    return lines


def generate_text_strip(text, seed, letter_size=64, pad=-18, space_w=30):
    if G is None or not text.strip(): return None
    tokens = tokenize(text)
    if not tokens: return None

    if seed >= 0: torch.manual_seed(seed)
    base_noise = torch.randn(1, LATENT_DIM, device=device)

    glyphs = []
    with torch.no_grad():
        for char_idx, (_, cid) in enumerate(tokens):
            if cid == -1:
                glyphs.append(None)
                continue
            noise = base_noise
            label = torch.tensor([cid], device=device)
            raw = (G(noise, label) + 1) / 2.0
            arr = (raw.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode='L')
            if letter_size != 64:
                pil = pil.resize((letter_size, letter_size), Image.LANCZOS)
            glyphs.append(pil)

    total_w = 0
    for i, g in enumerate(glyphs):
        if g is None:
            total_w += space_w
        else:
            total_w += g.width
            if i < len(glyphs) - 1 and glyphs[i + 1] is not None:
                total_w += pad
    total_w = max(letter_size, total_w)

    canvas = Image.new('L', (total_w, letter_size), 0)
    x = 0
    for i, g in enumerate(glyphs):
        if g is None:
            x += space_w
        else:
            tmp = Image.new('L', (total_w, letter_size), 0)
            tmp.paste(g, (x, 0))
            canvas = ImageChops.lighter(canvas, tmp)
            x += g.width
            if i < len(glyphs) - 1 and glyphs[i + 1] is not None:
                x += pad

    arr_final = np.array(canvas).astype(np.uint8)
    mask = _make_alpha_mask(arr_final)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return Image.fromarray(rgba, 'RGBA')


def generate_multiline_strip(text, seed, letter_size=64, pad=-18, space_w=30, wrap_width=0, line_gap=8):
    if wrap_width > 0:
        lines = _wrap_words(text, wrap_width, letter_size, pad, space_w)
    else:
        lines = [text]
    if not lines: return None
    strips = []
    max_w = 0
    for line in lines:
        s = generate_text_strip(line, seed, letter_size, pad, space_w)
        if s is not None:
            strips.append(s)
            max_w = max(max_w, s.width)
    if not strips: return None
    if len(strips) == 1: return strips[0]
    row_h = letter_size + line_gap
    total_h = row_h * len(strips) - line_gap
    canvas = Image.new('RGBA', (max_w, total_h), (0, 0, 0, 0))
    for i, s in enumerate(strips):
        canvas.paste(s, (0, i * row_h), s)
    return canvas


def prepare_background(bg_img, bg_mode, canvas_w, canvas_h, pan_x=0.5, pan_y=0.5):
    cw, ch = int(canvas_w), int(canvas_h)
    if bg_img is not None:
        img = bg_img.convert("RGBA")
        if "Cover" in bg_mode:
            return ImageOps.fit(img, (cw, ch), Image.LANCZOS, centering=(pan_x, pan_y))
        elif "Contain" in bg_mode:
            return ImageOps.pad(img, (cw, ch), Image.LANCZOS, color=(26, 26, 46, 255))
        else:
            return img.resize((cw, ch), Image.LANCZOS)
    else:
        return Image.new("RGBA", (cw, ch), (26, 26, 46, 255))


def render_canvas(layers, bg_img, bg_mode, pan_x, pan_y, canvas_w=1920, canvas_h=1080):
    base = prepare_background(bg_img, bg_mode, canvas_w, canvas_h, pan_x, pan_y)
        
    for block in layers:
        if not block.get("visible", True): continue
        strip = generate_multiline_strip(
            block["text"], block["seed"],
            letter_size=block.get("letter_size", 128),
            pad=block.get("pad", -18),
            space_w=block.get("space_w", 30),
            wrap_width=block.get("max_width", 0),
        )
        if strip is None: continue
        sc = block["scale"] / 100.0
        new_w = max(1, int(strip.width * sc))
        new_h = max(1, int(strip.height * sc))
        strip = strip.resize((new_w, new_h), Image.LANCZOS)
        rot = block.get("rotation", 0)
        if rot != 0:
            strip = strip.rotate(-rot, expand=True)
            
        col = block.get("color", "#ffffff")
        stroke_w = block.get("stroke_width", 0)
        stroke_c = block.get("stroke_color", "#000000")
        
        final_colored_strip, pad_offset = apply_stroke_and_color(strip, col, stroke_w, stroke_c)
        
        temp_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        paste_x = int(block["x"]) - pad_offset
        paste_y = int(block["y"]) - pad_offset
        
        temp_layer.paste(final_colored_strip, (paste_x, paste_y))
        base = Image.alpha_composite(base, temp_layer)
        
    return base.convert("RGB")


def build_gallery_images(preview_text, seeds=None):
    if seeds is None: seeds = GALLERY_SEEDS[:GALLERY_INIT_COUNT]
    results = []
    for seed in seeds:
        strip = generate_text_strip(preview_text, seed, letter_size=48, pad=-14, space_w=20)
        if strip is None:
            img = Image.new("RGB", (160, 56), "#1a1a1a")
            d = ImageDraw.Draw(img)
            d.text((10, 20), f"seed {seed}", fill="#555")
        else:
            bg = Image.new("RGB", (strip.width + 16, strip.height + 12), "#111111")
            colored = recolor(strip, "#ffffff")
            bg.paste(colored, (8, 6), strip)
            img = bg
        results.append((img, f"seed {seed}"))
    return results


def make_block(text, seed, color, letter_size, pad, space_w, x=40, y=40,
               scale=100, rotation=0, visible=True, max_width=0,
               stroke_width=0, stroke_color="#000000"):
    return {
        "text": text, "seed": seed, "color": color,
        "letter_size": letter_size, "pad": pad, "space_w": space_w,
        "x": x, "y": y, "scale": scale, "rotation": rotation,
        "visible": visible, "max_width": max_width,
        "stroke_width": stroke_width, "stroke_color": stroke_color
    }

# ── Pan Controls ──
def pan_left_fn(x, y): return max(0.0, float(x) - 0.05), float(y)
def pan_right_fn(x, y): return min(1.0, float(x) + 0.05), float(y)
def pan_up_fn(x, y): return float(x), max(0.0, float(y) - 0.05)
def pan_down_fn(x, y): return float(x), min(1.0, float(y) + 0.05)
def pan_center_fn(x, y): return 0.5, 0.5


# ════════════════════════════════════════════════════════════════════════════
# UI / CSS
# ════════════════════════════════════════════════════════════════════════════
CSS = """
body, .gradio-container { background: #0f0f17 !important; color: #e8e4dc !important; }
.gr-box, .gr-form { background: #16161f !important; border: 1px solid #2a2a3a !important; }
.gr-button-primary { background: #c8a96e !important; color: #0f0f17 !important;
                     font-weight: 700 !important; border: none !important; }
.gr-button-secondary { background: #1e1e2e !important; color: #c8a96e !important;
                        border: 1px solid #c8a96e !important; }
label, .gr-input-label { color: #9a9080 !important; font-size: 11px !important;
                          text-transform: uppercase; letter-spacing: .08em; }
input[type=number], input[type=text], textarea {
    background: #0f0f17 !important; color: #e8e4dc !important;
    border: 1px solid #2a2a3a !important; }
.gr-slider input[type=range] { accent-color: #c8a96e; }
#seed-gallery .gallery-item img { border: 2px solid transparent; border-radius: 4px;
                                   cursor: pointer; transition: border .15s; }
#seed-gallery .gallery-item img:hover { border-color: #c8a96e; }
#seed-gallery .gallery-item.selected img { border-color: #c8a96e; box-shadow: 0 0 8px #c8a96e88; }
#layers-panel { font-family: 'Courier New', monospace; font-size: 12px; }
#layers-panel textarea { background: #0b0b12 !important; }

#canvas-output img { border-radius: 6px; cursor: crosshair; max-height: 75vh !important; width: auto !important; max-width: 100% !important; margin: 0 auto; box-shadow: 0 4px 15px rgba(0,0,0,0.4); }

.section-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: .14em;
    color: #5a5060; border-bottom: 1px solid #2a2a3a;
    padding-bottom: 4px; margin: 10px 0 6px;
}

.canvas-instruction {
    text-align: center; color: #c8a96e; font-size: 13px; font-weight: 500;
    margin-bottom: 15px; padding: 8px; background: rgba(200, 169, 110, 0.1); 
    border-radius: 6px; border: 1px dashed rgba(200, 169, 110, 0.3);
}

.pan-btn { min-width: 40px !important; padding: 5px !important; font-size: 18px !important; }

/* 📱 MOBILE RESPONSIVE CSS */
@media (max-width: 768px) {
    .section-title { font-size: 12px !important; margin: 15px 0 8px !important; }
    input[type=number], input[type=text], textarea { font-size: 16px !important; }
    #canvas-output img { max-height: 65vh !important; }
    .gr-box { padding: 10px !important; }
    button { padding: 12px !important; }
    .pan-btn { padding: 10px !important; font-size: 20px !important; }
}
"""

with gr.Blocks(title="Հայ Ձեռագիր · Studio") as demo:

    gr.HTML(f"<style>{CSS}</style>", visible=False)

    layers_state   = gr.State([])
    chosen_seed    = gr.State(555)  
    visible_count  = gr.State(GALLERY_INIT_COUNT)

    gr.Markdown("## ✍️  Հայ Ձեռագիր Studio ")

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=260):

            gr.HTML('<div class="section-title">🎨 Ոճ — ընտրիր seed</div>')

            preview_txt = gr.Textbox(
                value=PREVIEW_TEXT, label="Preview տեքստ (gallery-ի համար)", lines=1
            )
            refresh_gallery_btn = gr.Button("🔄 Թարմացնել", size="sm", variant="secondary")

            seed_gallery = gr.Gallery(
                label="Սեղմիր ոճ ընտրելու → սերմ կհիշվի",
                columns=2, rows=2, height=320,
                elem_id="seed-gallery", show_label=True,
                object_fit="contain", allow_preview=False,
            )
            more_seeds_btn = gr.Button("➕ Ավելացնել seeds", size="sm", variant="secondary")
            
            chosen_seed_display = gr.Number(
                label="Ընտրված seed", value=555, interactive=True,
                minimum=0, maximum=99999,
            )

            gr.HTML('<div class="section-title">➕ Նոր բլոք</div>')

            new_text = gr.Textbox(
                label="Հայերեն տեքստ", placeholder="Բարև Աշխարհ", lines=1
            )
                
            with gr.Row():
                new_size = gr.Slider(32, 500, step=8, value=128, label="Տառի չափ")
                new_pad  = gr.Slider(-40, 10, step=1, value=-18, label="Տառամիջյան հեռավորություն")
            new_space = gr.Slider(5, 120, step=1, value=30, label="Բառերի հեռավորություն (Բացատ)")

            add_block_btn = gr.Button("➕ Ավելացնել բլոք", variant="primary", size="lg")

            gr.HTML('<div class="section-title">🖼 Ֆոն</div>')
            bg_input = gr.Image(label="Ֆոնի նկար (կամայական)", type="pil", height=120)
            
            bg_fit_mode = gr.Radio(
                choices=["Կտրել և լցնել (Cover)", "Տեղավորել (Contain)", "Ձգել (Stretch)"], 
                value="Կտրել և լցնել (Cover)", 
                label="Ֆոնի տեղադրում (Fit Mode)",
                interactive=True
            )
            
            with gr.Accordion("↔️ Ֆոնի Տեղաշարժ (միայն Cover ռեժիմում)", open=False):
                with gr.Row():
                    bg_pan_x = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Հորիզոնական (X)")
                    bg_pan_y = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Ուղղահայաց (Y)")
                with gr.Row():
                    pan_left = gr.Button("⬅️", elem_classes="pan-btn")
                    pan_right = gr.Button("➡️", elem_classes="pan-btn")
                    pan_down = gr.Button("⬆️", elem_classes="pan-btn")
                    pan_up = gr.Button("⬇️", elem_classes="pan-btn")
                    pan_center = gr.Button("⏺️", elem_classes="pan-btn")

        with gr.Column(scale=3):
            
            gr.HTML('<div class="canvas-instruction">📍 Սեղմեք նկարի վրա՝ ընտրված տեքստը (Layer) տեղափոխելու համար</div>')
            
            canvas_out = gr.Image(
                show_label=False,
                container=False,
                type="pil",
                elem_id="canvas-output", 
                interactive=False,
            )

            with gr.Row():
                render_btn  = gr.Button("🖼 Render", variant="primary")
                export_btn  = gr.Button("💾 Export PNG", variant="secondary")
                clear_btn   = gr.Button("🗑 Մաքրել", variant="secondary")

            export_file = gr.File(
                label="⬇ Ներբեռնել PNG", visible=False, interactive=False
            )
            canvas_info = gr.Textbox(label="Վիճակ", interactive=False, lines=1)

        with gr.Column(scale=1, min_width=260):

            gr.HTML('<div class="section-title">📚 Layers</div>')

            layers_display = gr.HTML(value="<i style='color:#555'>Դատարկ</i>",
                                     elem_id="layers-panel")

            sel_idx = gr.Number(label="Ընտրված Layer-ը #", value=0, minimum=0,
                                step=1, interactive=True)
            
            # Նոր ընտրության կոճակներ
            with gr.Row():
                sel_prev_btn = gr.Button("◀ Նախորդ Layer", size="sm")
                sel_next_btn = gr.Button("Հաջորդ Layer ▶", size="sm")
                
            gr.HTML('<div class="section-title">⇵ Շերտերի Դասավորություն (Z-Index)</div>')
            with gr.Row():
                layer_up_btn   = gr.Button("🔼 Բերել առաջ", size="sm", variant="secondary")
                layer_down_btn = gr.Button("🔽 Տանել հետ", size="sm", variant="secondary")
                layer_del_btn  = gr.Button("🗑 Ջնջել", size="sm", variant="secondary")

            gr.HTML('<div class="section-title">📍 Դիրք</div>')
            sel_x = gr.Slider(0, 5000, step=1, value=40, label="X (px) →")
            sel_y = gr.Slider(0, 5000, step=1, value=40, label="Y (px) ↓")

            gr.HTML('<div class="section-title">⚙️ Ընտրված բլոք</div>')

            sel_text    = gr.Textbox(label="Տեքստ", lines=1, interactive=True)
            
            sel_seed    = gr.Number(label="Seed", value=555, minimum=0,
                                    maximum=99999, step=1, interactive=True)
            
            with gr.Row():
                sel_visible = gr.Checkbox(label="Տեսանելի 👁", value=True)
                auto_color_btn = gr.Button("🎨 Ավտո Գույն", size="sm")
                
            sel_color = gr.ColorPicker(value="#ffffff", label="Տեքստի Գույն")
            
            with gr.Row():
                sel_stroke_w = gr.Slider(0, 50, step=1, value=0, label="Եզրագծի հաստություն (Stroke)")
                sel_stroke_c = gr.ColorPicker(value="#000000", label="Եզրագծի Գույն")
            
            with gr.Row():
                sel_size = gr.Slider(32, 500, step=4, value=128, label="Տառի չափ")
                sel_pad  = gr.Slider(-40, 10, step=1, value=-18, label="Տառամիջյան հեռավորություն")
            sel_space = gr.Slider(5, 120, step=1, value=30, label="Բառերի հեռավորություն (Բացատ)")

            sel_sc   = gr.Slider(10, 300, step=1, value=100, label="Scale (%)")
            sel_rot  = gr.Slider(-180, 180, step=1, value=0, label="Rotation (°)")
            sel_maxw = gr.Slider(0, 3000, step=10, value=0,
                                 label="Wrap լայնություն px (0 = չփաթաթել)")

            apply_transform_btn = gr.Button("✓ Կիրառել", variant="primary")

            gr.HTML('<div class="section-title">📐 Canvas չափ (Resolution)</div>')
            
            preset_choices = [
                "🖥️ FHD (16:9)",
                "📱 Story (9:16)",
                "📸 Insta (1:1)",
                "🖼️ Port. (4:5)",
                "📺 4K (16:9)",
                "⚙️ Custom"
            ]
            
            canvas_preset = gr.Radio(
                choices=preset_choices, 
                value="🖥️ FHD (16:9)", 
                label="Արագ ընտրություն (Aspect Ratios)", 
                interactive=True
            )
            
            with gr.Row():
                canvas_w = gr.Number(label="W (Լայնություն)", value=1920, minimum=100, maximum=8000, step=10)
                canvas_h = gr.Number(label="H (Բարձրություն)", value=1080, minimum=100, maximum=8000, step=10)

    # ── Event handlers ────────────────────────────────────────────────────────

    def apply_canvas_preset(preset, current_w, current_h):
        if "FHD" in preset: return 1920, 1080
        if "Story" in preset: return 1080, 1920
        if "Insta" in preset: return 1080, 1080
        if "Port." in preset: return 1080, 1350
        if "4K" in preset: return 3840, 2160
        return current_w, current_h

    def refresh_gallery(preview_text, count):
        seeds = GALLERY_SEEDS[:int(count)]
        return build_gallery_images(preview_text, seeds)

    def load_more_seeds(preview_text, count):
        import random
        current_count = int(count)
        new_count = current_count + 4
        while len(GALLERY_SEEDS) < new_count:
            GALLERY_SEEDS.append(random.randint(10000, 99999))
        seeds = GALLERY_SEEDS[:new_count]
        return build_gallery_images(preview_text, seeds), new_count

    def on_gallery_select(evt: gr.SelectData, count):
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        seeds = GALLERY_SEEDS[:int(count)]
        seed = seeds[idx % len(seeds)]
        return seed, seed

    def layers_to_html(layers, sel_idx=0):
        if not layers:
            return "<i style='color:#555;font-size:12px'>Դատարկ</i>"
        rows = ""
        for i, b in enumerate(layers):
            eye  = "👁" if b.get("visible", True) else "🚫"
            text = b["text"][:16]
            seed = b["seed"]
            
            is_sel = (i == int(sel_idx))
            indicator = "🔴" if is_sel else "&nbsp;&nbsp;"
            bg_color = "#3a2a2a" if is_sel else "transparent"
            border_color = "#c8a96e" if is_sel else "#2a2a3a"
            
            rows += (
                f"<tr style='border-bottom:1px solid {border_color}; background-color:{bg_color};'>"
                f"<td style='padding:4px 6px;color:#9a9080; font-weight:bold;'>{indicator} {i}</td>"
                f"<td style='padding:4px 6px;color:#e8e4dc'>{text}</td>"
                f"<td style='padding:4px 6px;color:#c8a96e'>{seed}</td>"
                f"<td style='padding:4px 6px'>{eye}</td>"
                f"</tr>"
            )
        return (
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;"
            "font-family:monospace;font-size:12px; min-width: 200px;'>"
            "<thead><tr style='color:#5a5060;border-bottom:1px solid #3a3a4a'>"
            "<th style='padding:2px 6px;text-align:left'>#</th>"
            "<th style='padding:2px 6px;text-align:left'>Տեքստ</th>"
            "<th style='padding:2px 6px;text-align:left'>Seed</th>"
            "<th style='padding:2px 6px;text-align:left'>👁</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</div>"
        )

    def compute_auto_color(base_image, cw_int, ch_int):
        roi_box = (0, 0, cw_int, max(1, int(ch_int * 0.3)))
        bg_roi = base_image.crop(roi_box)
        avg_brightness = calculate_brightness(bg_roi)
        
        if avg_brightness < 127:
            return "#ffffff"
        else:
            return "#000000"

    def add_block(layers, text, seed, size, pad, space, bg, bg_mode, pan_x, pan_y, cw, ch):
        try:
            if not text.strip():
                return (layers, layers_to_html(layers, len(layers)-1 if layers else 0), gr.update(), "⚠️ Տեքստ մուտքագրեք",
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
            
            cw_int, ch_int = int(cw), int(ch)
            base = prepare_background(bg, bg_mode, cw_int, ch_int, pan_x, pan_y)
            
            strip = generate_multiline_strip(text, seed, letter_size=int(size), pad=int(pad), space_w=int(space))
            w, h = (strip.width, strip.height) if strip else (100, 64)
            
            x_pos = max(0, (cw_int - w) // 2)
            y_pos = max(0, int(ch_int * 0.05))
            
            calculated_color = compute_auto_color(base, cw_int, ch_int)

            block = make_block(text, int(seed), calculated_color, int(size), int(pad), int(space),
                               x=x_pos, y=y_pos,
                               stroke_width=0, stroke_color="#000000")
            layers = layers + [block]
            b = block
            new_idx = len(layers) - 1
            
            img = render_canvas(layers, bg, bg_mode, pan_x, pan_y, cw_int, ch_int)
            
            return (layers, layers_to_html(layers, new_idx), new_idx, f"✅ Ավելացվեց: «{text}»",
                    b["text"], b["seed"], b.get("visible", True), b["color"], b["stroke_width"], b["stroke_color"],
                    b["letter_size"], b["pad"], b["space_w"],
                    b["x"], b["y"], b["scale"], b.get("rotation", 0), b.get("max_width", 0), img)
        except Exception as e:
            return (layers, layers_to_html(layers, len(layers)-1 if layers else 0), gr.update(), f"❌ Սխալ: {e}",
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    def trigger_auto_color(layers, idx, bg, bg_mode, pan_x, pan_y, cw, ch):
        try:
            if idx is None or not layers:
                return layers, gr.update(), gr.update(), "⚠️ Layer ընտրված չէ"
            idx = int(idx)
            
            cw_int, ch_int = int(cw), int(ch)
            base = prepare_background(bg, bg_mode, cw_int, ch_int, pan_x, pan_y)
            calculated_color = compute_auto_color(base, cw_int, ch_int)
            
            layers[idx]["color"] = calculated_color
            img = render_canvas(layers, bg, bg_mode, pan_x, pan_y, cw_int, ch_int)
            
            return layers, calculated_color, img, "✅ Ավտո Գույնը Կիրառվեց"
        except Exception as e:
            return layers, gr.update(), gr.update(), f"❌ Սխալ: {e}"

    def do_render(layers, bg, bg_mode, pan_x, pan_y, cw, ch):
        if not layers:
            return None, "⚠️ Ավելացրեք գոնե 1 բլոք"
        img = render_canvas(layers, bg, bg_mode, pan_x, pan_y, int(cw), int(ch))
        return img, f"✅ Թարմացվեց {int(cw)}×{int(ch)}"

    def do_export(layers, bg, bg_mode, pan_x, pan_y, cw, ch):
        import tempfile
        if not layers:
            return None, gr.update(visible=False), "⚠️ Ավելացրեք գոնե 1 բլոք"
        img = render_canvas(layers, bg, bg_mode, pan_x, pan_y, int(cw), int(ch))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, format="PNG")
        tmp.close()
        return img, gr.update(value=tmp.name, visible=True), f"✅ Export — {int(cw)}×{int(ch)}"

    def apply_transform(layers, idx, txt, seed, visible, color, stroke_w, stroke_c, size, pad, space, x, y, sc, rot, maxw, bg, bg_mode, pan_x, pan_y, cw, ch):
        try:
            if idx is None:
                return layers, layers_to_html(layers, 0), gr.update(), "⚠️ Layer ընտրված չէ"
            idx = int(idx)
            if not layers or idx >= len(layers) or idx < 0:
                return layers, layers_to_html(layers, idx), gr.update(), "⚠️ Layer գտնված չէ"
                
            b = dict(layers[idx])
            b["text"]         = txt.strip() if txt.strip() else b["text"]
            b["seed"]         = int(seed)
            b["visible"]      = bool(visible)
            b["color"]        = color
            b["stroke_width"] = int(stroke_w)
            b["stroke_color"] = stroke_c
            b["letter_size"]  = int(size)
            b["pad"]          = int(pad)
            b["space_w"]      = int(space)
            b["x"] = int(x); b["y"] = int(y)
            b["scale"]        = int(sc); b["rotation"] = int(rot)
            b["max_width"]    = int(maxw)
            
            layers = layers[:idx] + [b] + layers[idx+1:]
            
            img = render_canvas(layers, bg, bg_mode, pan_x, pan_y, int(cw), int(ch))
            return layers, layers_to_html(layers, idx), img, f"✓ Layer {idx} թարմացվեց"
        except Exception as e:
            idx_safe = int(idx) if idx is not None else 0
            return layers, layers_to_html(layers, idx_safe), gr.update(), f"❌ Սխալ: {e}"

    def on_canvas_click(evt: gr.SelectData, layers, idx, txt, seed, visible, color, stroke_w, stroke_c, size, pad, space, x, y, sc, rot, maxw, bg, bg_mode, pan_x, pan_y, cw, ch):
        try:
            if idx is None or not layers:
                return layers, layers_to_html(layers, 0), gr.update(), gr.update(), gr.update(), "⚠️ Layer ընտրված չէ"
            idx = int(idx)
            if idx >= len(layers) or idx < 0:
                return layers, layers_to_html(layers, idx), gr.update(), gr.update(), gr.update(), "⚠️ Layer գտնված չէ"
                
            click_x, click_y = evt.index
            
            strip = generate_multiline_strip(txt.strip() if txt.strip() else layers[idx]["text"], 
                                             int(seed), int(size), int(pad), int(space), int(maxw))
            if strip:
                sc_factor = int(sc) / 100.0
                new_w = max(1, int(strip.width * sc_factor))
                new_h = max(1, int(strip.height * sc_factor))
                
                final_x = max(0, click_x - new_w // 2)
                final_y = max(0, click_y - new_h // 2)
            else:
                final_x, final_y = click_x, click_y

            b = dict(layers[idx])
            b["x"] = final_x
            b["y"] = final_y
            
            layers = layers[:idx] + [b] + layers[idx+1:]
            img = render_canvas(layers, bg, bg_mode, pan_x, pan_y, int(cw), int(ch))
            
            return layers, layers_to_html(layers, idx), img, final_x, final_y, f"📍 Տեղափոխվեց ({final_x}, {final_y})"
        except Exception as e:
            idx_safe = int(idx) if idx is not None else 0
            return layers, layers_to_html(layers, idx_safe), gr.update(), gr.update(), gr.update(), f"❌ Սխալ: {e}"

    def get_prev_layer(layers, idx):
        if idx is None: return 0
        return max(0, int(idx) - 1)

    def get_next_layer(layers, idx):
        if idx is None: return 0
        if not layers: return 0
        return min(len(layers) - 1, int(idx) + 1)

    def move_layer(layers, idx, direction):
        try:
            if idx is None: return layers, layers_to_html(layers, 0), 0
            idx = int(idx)
            if direction == "up" and idx > 0:
                layers[idx], layers[idx-1] = layers[idx-1], layers[idx]
                idx -= 1
            elif direction == "down" and idx < len(layers)-1:
                layers[idx], layers[idx+1] = layers[idx+1], layers[idx]
                idx += 1
            return layers, layers_to_html(layers, idx), idx
        except Exception:
            return layers, layers_to_html(layers, 0), 0

    def delete_layer(layers, idx):
        try:
            if idx is None: return layers, layers_to_html(layers, 0), 0, "⚠️ Ընտրված չէ"
            idx = int(idx)
            if layers and 0 <= idx < len(layers):
                layers = layers[:idx] + layers[idx+1:]
            new_idx = max(0, idx-1) if layers else 0
            return layers, layers_to_html(layers, new_idx), new_idx, f"🗑 Layer {idx} ջնջված"
        except Exception:
            return layers, layers_to_html(layers, 0), 0, "❌ Սխալ"

    def clear_all():
        return [], layers_to_html([], 0), None, "🗑 Ամեն մաքրված"

    def update_sel_controls(layers, idx):
        try:
            if idx is None: idx = 0
            idx = int(idx)
            html = layers_to_html(layers, idx)
            if not layers or idx >= len(layers) or idx < 0:
                return html, "", 555, True, "#ffffff", 0, "#000000", 128, -18, 30, 40, 40, 100, 0, 0
            b = layers[idx]
            return (html, b["text"], b["seed"], b.get("visible", True), 
                    b.get("color", "#ffffff"), b.get("stroke_width", 0), b.get("stroke_color", "#000000"),
                    b.get("letter_size", 128), 
                    b.get("pad", -18), b.get("space_w", 30),
                    b["x"], b["y"], b["scale"], b.get("rotation", 0),
                    b.get("max_width", 0))
        except Exception as e:
            return layers_to_html(layers, 0), "", 555, True, "#ffffff", 0, "#000000", 128, -18, 30, 40, 40, 100, 0, 0

    # ── EVENT BINDINGS ────────────────────────────────────────────────────────

    _render_inputs = [layers_state, bg_input, bg_fit_mode, bg_pan_x, bg_pan_y, canvas_w, canvas_h]
    
    render_btn.click(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    export_btn.click(fn=do_export, inputs=_render_inputs, outputs=[canvas_out, export_file, canvas_info])
    clear_btn.click(fn=clear_all, outputs=[layers_state, layers_display, canvas_out, canvas_info])

    canvas_preset.change(
        fn=apply_canvas_preset, inputs=[canvas_preset, canvas_w, canvas_h], outputs=[canvas_w, canvas_h]
    ).then(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])

    bg_fit_mode.change(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    bg_input.change(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    bg_pan_x.change(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    bg_pan_y.change(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    canvas_w.change(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    canvas_h.change(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])

    pan_left.click(fn=pan_left_fn, inputs=[bg_pan_x, bg_pan_y], outputs=[bg_pan_x, bg_pan_y]).then(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    pan_right.click(fn=pan_right_fn, inputs=[bg_pan_x, bg_pan_y], outputs=[bg_pan_x, bg_pan_y]).then(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    pan_up.click(fn=pan_up_fn, inputs=[bg_pan_x, bg_pan_y], outputs=[bg_pan_x, bg_pan_y]).then(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    pan_down.click(fn=pan_down_fn, inputs=[bg_pan_x, bg_pan_y], outputs=[bg_pan_x, bg_pan_y]).then(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])
    pan_center.click(fn=pan_center_fn, inputs=[bg_pan_x, bg_pan_y], outputs=[bg_pan_x, bg_pan_y]).then(fn=do_render, inputs=_render_inputs, outputs=[canvas_out, canvas_info])

    refresh_gallery_btn.click(fn=refresh_gallery, inputs=[preview_txt, visible_count], outputs=[seed_gallery])
    more_seeds_btn.click(fn=load_more_seeds, inputs=[preview_txt, visible_count], outputs=[seed_gallery, visible_count])
    seed_gallery.select(fn=on_gallery_select, inputs=[visible_count], outputs=[chosen_seed, chosen_seed_display])
    chosen_seed_display.change(fn=lambda v: int(v), inputs=[chosen_seed_display], outputs=[chosen_seed])

    _full_transform_inputs = [layers_state, sel_idx,
                              sel_text, sel_seed, sel_visible, sel_color, sel_stroke_w, sel_stroke_c,
                              sel_size, sel_pad, sel_space,
                              sel_x, sel_y, sel_sc, sel_rot, sel_maxw,
                              bg_input, bg_fit_mode, bg_pan_x, bg_pan_y, canvas_w, canvas_h]
    _transform_outputs = [layers_state, layers_display, canvas_out, canvas_info]

    add_block_btn.click(fn=add_block,
                         inputs=[layers_state, new_text, chosen_seed,
                                 new_size, new_pad, new_space,
                                 bg_input, bg_fit_mode, bg_pan_x, bg_pan_y, canvas_w, canvas_h],
                         outputs=[layers_state, layers_display, sel_idx, canvas_info,
                                  sel_text, sel_seed, sel_visible, sel_color, sel_stroke_w, sel_stroke_c,
                                  sel_size, sel_pad, sel_space,
                                  sel_x, sel_y, sel_sc, sel_rot, sel_maxw, canvas_out])

    auto_color_btn.click(fn=trigger_auto_color, 
                         inputs=[layers_state, sel_idx, bg_input, bg_fit_mode, bg_pan_x, bg_pan_y, canvas_w, canvas_h],
                         outputs=[layers_state, sel_color, canvas_out, canvas_info])

    apply_transform_btn.click(fn=apply_transform, inputs=_full_transform_inputs, outputs=_transform_outputs)

    for ctrl in [sel_text, sel_seed, sel_visible, sel_color, sel_stroke_c, sel_stroke_w, sel_size, sel_pad, sel_space, sel_x, sel_y, sel_sc, sel_rot, sel_maxw]:
        try:
            ctrl.input(fn=apply_transform, inputs=_full_transform_inputs, outputs=_transform_outputs)
        except Exception:
            ctrl.change(fn=apply_transform, inputs=_full_transform_inputs, outputs=_transform_outputs)

    canvas_out.select(fn=on_canvas_click, inputs=_full_transform_inputs, outputs=[layers_state, layers_display, canvas_out, sel_x, sel_y, canvas_info])

    # Ընտրել Նախորդ/Հաջորդ
    sel_prev_btn.click(fn=get_prev_layer, inputs=[layers_state, sel_idx], outputs=[sel_idx])
    sel_next_btn.click(fn=get_next_layer, inputs=[layers_state, sel_idx], outputs=[sel_idx])

    # Z-Index Փոփոխություններ
    layer_up_btn.click(fn=lambda l,i: move_layer(l,i,"up"), inputs=[layers_state, sel_idx], outputs=[layers_state, layers_display, sel_idx])
    layer_down_btn.click(fn=lambda l,i: move_layer(l,i,"down"), inputs=[layers_state, sel_idx], outputs=[layers_state, layers_display, sel_idx])
    layer_del_btn.click(fn=delete_layer, inputs=[layers_state, sel_idx], outputs=[layers_state, layers_display, sel_idx, canvas_info])

    # Շերտի ընտրության ժամանակ UI թարմացում
    sel_idx.change(fn=update_sel_controls, inputs=[layers_state, sel_idx],
                    outputs=[layers_display, sel_text, sel_seed, sel_visible, sel_color, sel_stroke_w, sel_stroke_c,
                             sel_size, sel_pad, sel_space,
                             sel_x, sel_y, sel_sc, sel_rot, sel_maxw])

    demo.load(fn=lambda: build_gallery_images(PREVIEW_TEXT, GALLERY_SEEDS[:GALLERY_INIT_COUNT]), outputs=[seed_gallery])

if __name__ == "__main__":
    demo.launch(share=True)
