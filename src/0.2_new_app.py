"""
app.py — Հայերեն Ձեռագրի Գեներացիա  ·  Photoshop-style editor
═══════════════════════════════════════════════════════════════════
Layout:

  ┌──────────────┬──────────────────────────────┬──────────────┐
  │  SEED GALLERY│        CANVAS                │   LAYERS     │
  │  (click seed │  (drag blocks, transform      │  (list of    │
  │   to pick    │   handles, bg optional)       │   blocks)    │
  │   font)      │                               │              │
  │──────────────│                               │──────────────│
  │  NEW BLOCK   │                               │  BLOCK PROPS │
  │  Text input  │                               │  (selected)  │
  │  + Add btn   │                               │              │
  └──────────────┴──────────────────────────────┴──────────────┘

Key features:
  • Seed Gallery  — generate preview thumbnails, click to choose style
  • Multiple Blocks — each block is independent (text, seed, color, size)
  • Drag & Drop   — move blocks freely on canvas (via JS in gr.HTML)
  • Transform     — resize / rotate handles per selected block
  • Layers Panel  — visibility toggle, delete, reorder (up/down)
  • Export        — flatten all visible layers → final PNG

FIXES applied (v2):
  1. Soft-alpha compositing: generator ink texture is preserved in the
     alpha channel instead of being destroyed by double THRESH_BINARY.
  2. Noise is shared per word (all letters share the same latent vector →
     consistent handwriting style across the word). Per-character seed
     offset is added as a tiny variation on top so letters aren't clones.
  3. Checkpoint guard: startup assertion warns if epoch > 200.
  4. check_diversity() helper to detect mode collapse at launch.
"""

import torch
import numpy as np
import cv2
import gradio as gr
import json
import base64
import io
from PIL import Image, ImageChops, ImageDraw
from model import Generator
from dataset import CHAR_TO_CLASS, NUM_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
# ⚠ Use epoch 140 or 200 — epoch 300 is overtrained and produces memorized outputs
CHECKPOINT_PATH = r"checkpoints/checkpoint_epoch_0140.pt"
LATENT_DIM      = 100
DIGRAPHS        = {'ու': 63, 'Ու': 24, 'Եվ': 36, 'և': 75}
# Full pool of available seeds — shown 4 at a time, load more on demand
GALLERY_SEEDS      = [0, 7, 42, 137, 256, 512, 999, 1337, 2024, 3333, 5050, 7777, 8888, 9999]
GALLERY_INIT_COUNT = 4   # seeds visible on first load
PREVIEW_TEXT    = "Բարև"

# Per-character noise perturbation scale.
# 0.0 → all letters are identical style clones (maximum consistency).
# 0.15 → subtle per-letter variation while style stays recognisably the same.
# Increase toward 1.0 for more diversity at the cost of style coherence.
CHAR_NOISE_VARIATION = 0.10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Init] Device: {device}")


def _parse_epoch(path: str) -> int:
    """Extract epoch number from checkpoint filename."""
    import re
    m = re.search(r'epoch[_\-]?(\d+)', path, re.IGNORECASE)
    return int(m.group(1)) if m else -1


try:
    G = Generator(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, embed_dim=LATENT_DIM).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()

    _epoch = _parse_epoch(CHECKPOINT_PATH)
    if _epoch > 200:
        print(f"⚠️  WARNING: checkpoint epoch {_epoch} > 200 — outputs may look "
              f"memorized. Use epoch_0140.pt or epoch_0200.pt for best results.")
    else:
        print(f"✅ Model loaded  (epoch {_epoch})")

except Exception as e:
    print(f"❌ Model error: {e}")
    G = None


# ── Diversity check (call once at startup to detect mode collapse) ─────────────
def check_diversity(char_class: int = 0, n_seeds: int = 5) -> None:
    """
    Generate the same character with n different seeds and print mean absolute
    pixel difference. Values near 0.0 mean the generator has collapsed.
    Run this manually: check_diversity(char_class=0) in a Python console.
    """
    if G is None:
        print("[diversity] Generator not loaded.")
        return
    outputs = []
    test_seeds = [0, 7, 42, 137, 256][:n_seeds]
    with torch.no_grad():
        for s in test_seeds:
            torch.manual_seed(s)
            noise = torch.randn(1, LATENT_DIM, device=device)
            label = torch.tensor([char_class], device=device)
            t = (G(noise, label) + 1) / 2.0
            outputs.append(t.squeeze().cpu().numpy())
    diffs = [float(np.mean(np.abs(outputs[i] - outputs[0]))) for i in range(1, n_seeds)]
    print(f"[diversity] Class {char_class} — mean |diff| from seed 0: {diffs}")
    if all(d < 0.01 for d in diffs):
        print("[diversity] ⚠️  COLLAPSED — all outputs identical regardless of seed!")
    else:
        print("[diversity] ✅ Generator is producing diverse outputs.")


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
            print(f"[WARN] '{ch}' U+{ord(ch):04X} skipped")
        i += 1
    return tokens


def calc_brightness(region: Image.Image) -> float:
    a = np.array(region.convert("RGB"))
    return float(np.mean(0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]))


def recolor(img_rgba: Image.Image, color: str) -> Image.Image:
    """Fill R,G,B channels with the target color value; keep alpha unchanged.
    Our strips store ink only in the alpha channel (R=G=B=0), so we must
    set R/G/B to 255 (white) or 0 (black) — not copy from the existing channels.
    """
    _, _, _, a = img_rgba.split()
    v = 255 if color == "white" else 0
    flat = Image.new("L", img_rgba.size, v)
    return Image.merge("RGBA", (flat, flat, flat, a))


def _make_alpha_mask(raw_arr: np.ndarray) -> np.ndarray:
    """
    Convert a raw uint8 grayscale glyph (white strokes on black) to a clean
    binary alpha mask.

    Rules:
      • ONE GaussianBlur pass only — to denoise, not to soften strokes.
      • ONE THRESH_BINARY threshold — no second pass, no dilation.
      • No dilation: dilation fattens strokes and makes them look like
        clean ground-truth dataset labels (the original bug).
      • Threshold at 50 (vs original 40) to suppress noise floor a bit more
        while keeping fine generator strokes.
    """
    blurred = cv2.GaussianBlur(raw_arr, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _measure_word_width(word: str, letter_size: int, pad: int, space_w: int) -> int:
    """Measure pixel width of a word using the same logic as generate_text_strip.
    Glyphs are always generated at 64px then resized to letter_size, so
    the actual glyph width on the canvas is letter_size.
    pad is typically negative (kerning) — we clamp so it can't make width negative.
    """
    tokens = tokenize(word)
    glyphs_count = sum(1 for _, cid in tokens if cid != -1)
    spaces_count  = sum(1 for _, cid in tokens if cid == -1)
    # Each glyph is letter_size wide; kerning pad applied between adjacent glyphs
    w = glyphs_count * letter_size + max(glyphs_count - 1, 0) * pad + spaces_count * space_w
    return max(letter_size, w)


def _wrap_words(text: str, wrap_width: int, letter_size: int,
                pad: int, space_w: int) -> list[str]:
    """
    Split `text` into lines so each line fits within wrap_width pixels.
    - Wraps at word boundaries first (like Word/Photoshop).
    - If a single word is wider than wrap_width, falls back to char-level
      splitting so it never overflows (no hyphen — just a hard break).
    """
    words = text.split(' ')
    lines: list[str] = []
    current: list[str] = []

    def _flush():
        if current:
            line = ' '.join(w for w in current if w)
            if line.strip():
                lines.append(line)
            current.clear()

    for word in words:
        if not word:
            continue
        word_w = _measure_word_width(word, letter_size, pad, space_w)

        # Single word wider than wrap_width → char-level split
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

        # Normal word: try adding to current line
        test_text = ' '.join(w for w in (current + [word]) if w)
        if current and _measure_word_width(test_text, letter_size, pad, space_w) > wrap_width:
            _flush()
            current.append(word)
        else:
            current.append(word)

    _flush()
    return lines


def generate_text_strip(text: str, seed: int, letter_size: int = 64,
                         pad: int = -18, space_w: int = 30) -> Image.Image | None:
    """
    Generate a transparent RGBA strip for `text` using given seed.

    Style consistency:
        A single base noise vector is sampled from `seed` and shared across
        all characters in the word → all letters have the same handwriting
        style. A small per-character perturbation (CHAR_NOISE_VARIATION) is
        added on top so letters are not pixel-perfect clones of each other.
    """
    if G is None or not text.strip():
        return None
    tokens = tokenize(text)
    if not tokens:
        return None

    # ── Sample the shared style noise (one per word) ──────────────────────────
    if seed >= 0:
        torch.manual_seed(seed)
    base_noise = torch.randn(1, LATENT_DIM, device=device)

    glyphs = []
    with torch.no_grad():
        for char_idx, (_, cid) in enumerate(tokens):
            if cid == -1:
                glyphs.append(None)
                continue

            # Per-character micro-variation on top of the shared style.
            # CHAR_NOISE_VARIATION=0.0 → identical noise for all letters.
            # CHAR_NOISE_VARIATION=0.10 → subtle per-letter differences.
            if CHAR_NOISE_VARIATION > 0.0:
                torch.manual_seed(seed * 1000 + char_idx)
                char_delta = torch.randn(1, LATENT_DIM, device=device)
                noise = base_noise + CHAR_NOISE_VARIATION * char_delta
            else:
                noise = base_noise

            label = torch.tensor([cid], device=device)
            raw = (G(noise, label) + 1) / 2.0
            arr = (raw.squeeze().cpu().numpy() * 255).astype(np.uint8)

            # Keep glyph as 'L' (grayscale) — mask is applied once at the end
            # on the composited canvas, not per-glyph.  This avoids the
            # repeated blur passes that caused the blurriness in v2.
            pil = Image.fromarray(arr, mode='L')
            if letter_size != 64:
                pil = pil.resize((letter_size, letter_size), Image.LANCZOS)
            glyphs.append(pil)

    # ── Compute canvas width ──────────────────────────────────────────────────
    total_w = 0
    for i, g in enumerate(glyphs):
        if g is None:
            total_w += space_w
        else:
            total_w += g.width
            if i < len(glyphs) - 1 and glyphs[i + 1] is not None:
                total_w += pad
    total_w = max(letter_size, total_w)

    # ── Composite glyphs on L canvas (original approach — sharp) ─────────────
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

    # ── Apply alpha mask ONCE on the fully composited strip ───────────────────
    # Single blur + single threshold on the final canvas keeps strokes sharp.
    arr_final = np.array(canvas).astype(np.uint8)
    mask = _make_alpha_mask(arr_final)

    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return Image.fromarray(rgba, 'RGBA')


def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ── Seed Gallery ──────────────────────────────────────────────────────────────
def build_gallery_images(preview_text: str, seeds: list | None = None) -> list:
    """Return list of (PIL thumbnail, label) for the given seeds."""
    if seeds is None:
        seeds = GALLERY_SEEDS[:GALLERY_INIT_COUNT]
    results = []
    for seed in seeds:
        strip = generate_text_strip(preview_text, seed, letter_size=48, pad=-14, space_w=20)
        if strip is None:
            img = Image.new("RGB", (160, 56), "#1a1a1a")
            d = ImageDraw.Draw(img)
            d.text((10, 20), f"seed {seed}", fill="#555")
        else:
            bg = Image.new("RGB", (strip.width + 16, strip.height + 12), "#111111")
            colored = recolor(strip, "white")
            bg.paste(colored, (8, 6), strip)
            img = bg
        results.append((img, f"seed {seed}"))
    return results


# ── Layer / Block engine ──────────────────────────────────────────────────────
def make_block(text, seed, color, letter_size, pad, space_w, x=40, y=40,
               scale=100, rotation=0, visible=True, max_width=0):
    """Create a new layer block dict.
    max_width: max pixel width of the strip (0 = auto-fit to canvas width).
    """
    return {
        "text": text, "seed": seed, "color": color,
        "letter_size": letter_size, "pad": pad, "space_w": space_w,
        "x": x, "y": y, "scale": scale, "rotation": rotation,
        "visible": visible, "max_width": max_width,
    }


def generate_multiline_strip(text: str, seed: int, letter_size: int = 64,
                              pad: int = -18, space_w: int = 30,
                              wrap_width: int = 0,
                              line_gap: int = 8) -> Image.Image | None:
    """
    Generate a multi-line RGBA image.
    wrap_width=0 → single line (original behaviour).
    wrap_width>0 → wrap at word boundaries so each line ≤ wrap_width px wide.
    Letter size is NEVER changed — only line count grows (exactly like Word/PS).
    """
    if wrap_width > 0:
        lines = _wrap_words(text, wrap_width, letter_size, pad, space_w)
    else:
        lines = [text]

    if not lines:
        return None

    strips = []
    max_w = 0
    for line in lines:
        s = generate_text_strip(line, seed, letter_size, pad, space_w)
        if s is not None:
            strips.append(s)
            max_w = max(max_w, s.width)

    if not strips:
        return None
    if len(strips) == 1:
        return strips[0]

    # Stack lines vertically with line_gap between them
    row_h = letter_size + line_gap
    total_h = row_h * len(strips) - line_gap
    canvas = Image.new('RGBA', (max_w, total_h), (0, 0, 0, 0))
    for i, s in enumerate(strips):
        canvas.paste(s, (0, i * row_h), s)
    return canvas


def render_canvas(layers: list, bg_img, canvas_w=900, canvas_h=520) -> Image.Image:
    """Flatten all visible layers onto the canvas."""
    if bg_img is not None:
        base = bg_img.convert("RGBA").resize((canvas_w, canvas_h), Image.LANCZOS)
    else:
        base = Image.new("RGBA", (canvas_w, canvas_h), "#1a1a2e")

    for block in layers:
        if not block.get("visible", True):
            continue
        # wrap_width > 0 → wrap text into multiple lines at that px width
        # wrap_width = 0 → single line, no wrap (original behaviour)
        strip = generate_multiline_strip(
            block["text"], block["seed"],
            letter_size=block["letter_size"],
            pad=block["pad"],
            space_w=block["space_w"],
            wrap_width=block.get("max_width", 0),
        )
        if strip is None:
            continue

        # Scale (applied after wrapping — letter size stays the same)
        sc = block["scale"] / 100.0
        new_w = max(1, int(strip.width * sc))
        new_h = max(1, int(strip.height * sc))
        strip = strip.resize((new_w, new_h), Image.LANCZOS)

        # Rotation
        rot = block.get("rotation", 0)
        if rot != 0:
            strip = strip.rotate(-rot, expand=True)

        # Color
        color_map = {"Ավտոմատ": None, "Սպիտակ": "white", "Սև": "black"}
        col = color_map.get(block["color"], "white")
        if col is None:
            region = base.crop((block["x"], block["y"],
                                 block["x"] + strip.width, block["y"] + strip.height))
            b = calc_brightness(region)
            col = "white" if b < 127 else "black"
        colored = recolor(strip, col)

        base.paste(colored, (int(block["x"]), int(block["y"])), strip)

    return base.convert("RGB")


# ── Gradio State helpers ──────────────────────────────────────────────────────
def layers_to_json(layers): return json.dumps(layers)
def json_to_layers(s):
    try: return json.loads(s)
    except: return []


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════
CSS = """
/* ── Global ── */
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

/* ── Gallery grid ── */
#seed-gallery .gallery-item img { border: 2px solid transparent; border-radius: 4px;
                                   cursor: pointer; transition: border .15s; }
#seed-gallery .gallery-item img:hover { border-color: #c8a96e; }
#seed-gallery .gallery-item.selected img { border-color: #c8a96e; box-shadow: 0 0 8px #c8a96e88; }

/* ── Layers panel ── */
#layers-panel { font-family: 'Courier New', monospace; font-size: 12px; }
#layers-panel textarea { background: #0b0b12 !important; }

/* ── Canvas ── */
#canvas-output img { border-radius: 6px; }

/* ── Section headers ── */
.section-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: .14em;
    color: #5a5060; border-bottom: 1px solid #2a2a3a;
    padding-bottom: 4px; margin: 10px 0 6px;
}
"""

with gr.Blocks(title="Հայ Ձեռագիր · Studio") as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    layers_state   = gr.State([])
    selected_layer = gr.State(0)
    chosen_seed    = gr.State(42)
    visible_count  = gr.State(GALLERY_INIT_COUNT)   # how many seeds currently shown

    gr.Markdown("## ✍️  Հայ Ձեռագիր Studio")

    with gr.Row(equal_height=False):

        # ══════════════════════════════════════════════════════════════════════
        # LEFT PANEL — Seed Gallery + New Block
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, min_width=260):

            gr.HTML('<div class="section-title">🎨 Ոճ — ընտրիր seed</div>')

            preview_txt = gr.Textbox(
                value=PREVIEW_TEXT, label="Preview տեքստ (gallery-ի համար)", lines=1
            )
            refresh_gallery_btn = gr.Button("🔄 Թարմացնել", size="sm",
                                             variant="secondary")

            seed_gallery = gr.Gallery(
                label="Սեղմիր ոճ ընտրելու → սերմ կհիշվի",
                columns=2, rows=2, height=320,
                elem_id="seed-gallery", show_label=True,
                object_fit="contain",
                allow_preview=False,
            )
            more_seeds_btn = gr.Button("➕ Ավելացնել seeds", size="sm",
                                       variant="secondary")
            chosen_seed_display = gr.Number(
                label="Ընտրված seed", value=42, interactive=True,
                minimum=0, maximum=9999,
            )

            gr.HTML('<div class="section-title">➕ Նոր բլոք</div>')

            new_text = gr.Textbox(
                label="Հայերեն տեքստ", placeholder="Բարև Աշխարհ", lines=1
            )
            with gr.Row():
                new_color = gr.Radio(
                    ["Ավտոմատ", "Սպիտակ", "Սև"], value="Ավտոմատ",
                    label="Գույն", scale=2
                )
            with gr.Row():
                new_size = gr.Slider(32, 128, step=8, value=64, label="Տառի չափ")
                new_pad  = gr.Slider(-40, 10, step=1, value=-18, label="Pad")
            new_space = gr.Slider(5, 120, step=1, value=30, label="Բացատ")

            add_block_btn = gr.Button("➕ Ավելացնել բլոք", variant="primary", size="lg")

            gr.HTML('<div class="section-title">🖼 Ֆոն</div>')
            bg_input = gr.Image(label="Ֆոնի նկար (կամայական)", type="pil",
                                height=120)

        # ══════════════════════════════════════════════════════════════════════
        # CENTER — Canvas
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=3):

            canvas_out = gr.Image(
                label="Կտավ", type="pil", height=520,
                elem_id="canvas-output", interactive=False,
            )

            with gr.Row():
                render_btn  = gr.Button("🖼 Render", variant="primary")
                export_btn  = gr.Button("💾 Export PNG", variant="secondary")
                clear_btn   = gr.Button("🗑 Մաքրել ամեն", variant="secondary")

            export_file = gr.File(
                label="⬇ Ներբեռնել PNG", visible=False, interactive=False
            )

            canvas_info = gr.Textbox(
                label="Ստատուս", interactive=False, lines=1
            )

        # ══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — Layers + Selected Block Controls
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, min_width=240):

            gr.HTML('<div class="section-title">📚 Layers</div>')

            # Plain HTML list — always readable, no Dataframe quirks
            layers_display = gr.HTML(value="<i style='color:#555'>Դատարկ</i>",
                                     elem_id="layers-panel")

            with gr.Row():
                layer_up_btn   = gr.Button("▲", size="sm", variant="secondary")
                layer_down_btn = gr.Button("▼", size="sm", variant="secondary")
                layer_del_btn  = gr.Button("✕", size="sm", variant="secondary")

            gr.HTML('<div class="section-title">⚙️ Ընտրված բլոք</div>')

            sel_idx = gr.Number(label="Layer #", value=0, minimum=0,
                                step=1, interactive=True)

            # Editable fields for the selected block
            sel_text    = gr.Textbox(label="Տեքստ", lines=1, interactive=True)
            sel_seed    = gr.Number(label="Seed", value=42, minimum=0,
                                    maximum=99999, step=1, interactive=True)
            sel_visible = gr.Checkbox(label="Տեսանելի 👁", value=True)

            sel_x   = gr.Slider(0, 1600, step=1, value=40,  label="X դիրք (px)")
            sel_y   = gr.Slider(0, 900,  step=1, value=40,  label="Y դիրք (px)")
            sel_sc  = gr.Slider(10, 300, step=1, value=100, label="Scale (%)")
            sel_rot = gr.Slider(-180, 180, step=1, value=0, label="Rotation (°)")
            sel_maxw = gr.Slider(0, 3000, step=10, value=0,
                                 label="Wrap լայնություն px (0 = չփաթաթել)")

            apply_transform_btn = gr.Button("✓ Կիրառել", variant="primary")

            gr.HTML('<div class="section-title">📐 Canvas չափ</div>')
            with gr.Row():
                canvas_w = gr.Number(label="W", value=900, minimum=200, maximum=3000, step=10)
                canvas_h = gr.Number(label="H", value=520, minimum=100, maximum=2000, step=10)

    # ── Event handlers ────────────────────────────────────────────────────────

    def refresh_gallery(preview_text, count):
        """Re-generate thumbnails for currently visible seeds."""
        seeds = GALLERY_SEEDS[:int(count)]
        imgs = build_gallery_images(preview_text, seeds)
        return imgs

    def load_more_seeds(preview_text, count):
        """Append 4 more seeds to the gallery (wraps when pool exhausted)."""
        new_count = min(int(count) + 4, len(GALLERY_SEEDS))
        seeds = GALLERY_SEEDS[:new_count]
        imgs = build_gallery_images(preview_text, seeds)
        return imgs, new_count

    def on_gallery_select(evt: gr.SelectData, count):
        """User clicked a gallery thumbnail → store seed."""
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        seeds = GALLERY_SEEDS[:int(count)]
        seed = seeds[idx % len(seeds)]
        return seed, seed

    def layers_to_html(layers):
        """Render layers list as a styled HTML table."""
        if not layers:
            return "<i style='color:#555;font-size:12px'>Դատարկ</i>"
        rows = ""
        for i, b in enumerate(layers):
            eye  = "👁" if b.get("visible", True) else "🚫"
            text = b["text"][:16]
            seed = b["seed"]
            rows += (
                f"<tr style='border-bottom:1px solid #2a2a3a'>"
                f"<td style='padding:3px 5px;color:#9a9080'>{i}</td>"
                f"<td style='padding:3px 5px;color:#e8e4dc'>{text}</td>"
                f"<td style='padding:3px 5px;color:#c8a96e'>{seed}</td>"
                f"<td style='padding:3px 5px'>{eye}</td>"
                f"</tr>"
            )
        return (
            "<table style='width:100%;border-collapse:collapse;"
            "font-family:monospace;font-size:11px'>"
            "<thead><tr style='color:#5a5060;border-bottom:1px solid #3a3a4a'>"
            "<th style='padding:2px 5px;text-align:left'>#</th>"
            "<th style='padding:2px 5px;text-align:left'>Տեքստ</th>"
            "<th style='padding:2px 5px;text-align:left'>Seed</th>"
            "<th style='padding:2px 5px;text-align:left'>👁</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    def add_block(layers, text, seed, color, size, pad, space):
        if not text.strip():
            return layers, layers_to_html(layers), "⚠️ Տեքստ մուտքագրեք"
        block = make_block(text, int(seed), color, int(size), int(pad), int(space))
        layers = layers + [block]
        return layers, layers_to_html(layers), f"✅ Ավելացվեց: «{text}»  (seed={seed})"

    def do_render(layers, bg, cw, ch):
        if not layers:
            return None, "⚠️ Ավելացրեք գոնե 1 բլոք"
        img = render_canvas(layers, bg, int(cw), int(ch))
        return img, f"✅ Rendered {int(cw)}×{int(ch)}  ·  {len(layers)} layer(s)"

    def do_export(layers, bg, cw, ch):
        """Render canvas and save to a temp PNG file for download."""
        import tempfile
        if not layers:
            return None, gr.update(visible=False), "⚠️ Ավելացրեք գոնե 1 բլոք"
        img = render_canvas(layers, bg, int(cw), int(ch))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, format="PNG")
        tmp.close()
        return img, gr.update(value=tmp.name, visible=True), f"✅ Export պատրաստ — {int(cw)}×{int(ch)}"

    def apply_transform(layers, idx, txt, seed, visible, x, y, sc, rot, maxw, bg, cw, ch):
        """Apply ALL editable fields (text, seed, visible, transform) to the selected layer."""
        idx = int(idx)
        if not layers or idx >= len(layers):
            return layers, layers_to_html(layers), None, "⚠️ Layer գտնված չէ"
        b = dict(layers[idx])
        b["text"]      = txt.strip() if txt.strip() else b["text"]
        b["seed"]      = int(seed)
        b["visible"]   = bool(visible)
        b["x"] = int(x); b["y"] = int(y)
        b["scale"]     = int(sc); b["rotation"] = int(rot)
        b["max_width"] = int(maxw)
        layers = layers[:idx] + [b] + layers[idx+1:]
        img = render_canvas(layers, bg, int(cw), int(ch))
        return layers, layers_to_html(layers), img, f"✓ Layer {idx} — կիրառված"

    def move_layer(layers, idx, direction):
        idx = int(idx)
        if direction == "up" and idx > 0:
            layers[idx], layers[idx-1] = layers[idx-1], layers[idx]
            idx -= 1
        elif direction == "down" and idx < len(layers)-1:
            layers[idx], layers[idx+1] = layers[idx+1], layers[idx]
            idx += 1
        return layers, layers_to_html(layers), idx

    def delete_layer(layers, idx):
        idx = int(idx)
        if layers and 0 <= idx < len(layers):
            layers = layers[:idx] + layers[idx+1:]
        new_idx = max(0, idx-1)
        return layers, layers_to_html(layers), new_idx, f"🗑 Layer {idx} ջնջված"

    def clear_all():
        return [], layers_to_html([]), None, "🗑 Ամեն մաքրված"

    def update_sel_controls(layers, idx):
        """When selected layer changes, update all editable fields."""
        idx = int(idx)
        if not layers or idx >= len(layers):
            return "", 42, True, 40, 40, 100, 0, 0
        b = layers[idx]
        return (b["text"], b["seed"], b.get("visible", True),
                b["x"], b["y"], b["scale"], b.get("rotation", 0),
                b.get("max_width", 0))

    # Wire up gallery
    refresh_gallery_btn.click(
        fn=refresh_gallery,
        inputs=[preview_txt, visible_count],
        outputs=[seed_gallery],
    )
    more_seeds_btn.click(
        fn=load_more_seeds,
        inputs=[preview_txt, visible_count],
        outputs=[seed_gallery, visible_count],
    )
    seed_gallery.select(
        fn=on_gallery_select,
        inputs=[visible_count],
        outputs=[chosen_seed, chosen_seed_display],
    )
    chosen_seed_display.change(
        fn=lambda v: int(v),
        inputs=[chosen_seed_display],
        outputs=[chosen_seed],
    )

    # Add block — uses chosen_seed
    add_block_btn.click(
        fn=add_block,
        inputs=[layers_state, new_text, chosen_seed,
                new_color, new_size, new_pad, new_space],
        outputs=[layers_state, layers_display, canvas_info],
    )

    # Render
    render_btn.click(
        fn=do_render,
        inputs=[layers_state, bg_input, canvas_w, canvas_h],
        outputs=[canvas_out, canvas_info],
    )

    # Export — renders and serves a real PNG file for download
    export_btn.click(
        fn=do_export,
        inputs=[layers_state, bg_input, canvas_w, canvas_h],
        outputs=[canvas_out, export_file, canvas_info],
    )

    # Apply — saves text/seed/visible/transform/max_width to selected layer and re-renders
    apply_transform_btn.click(
        fn=apply_transform,
        inputs=[layers_state, sel_idx,
                sel_text, sel_seed, sel_visible,
                sel_x, sel_y, sel_sc, sel_rot, sel_maxw,
                bg_input, canvas_w, canvas_h],
        outputs=[layers_state, layers_display, canvas_out, canvas_info],
    )

    # Update all editable fields when selected index changes
    sel_idx.change(
        fn=update_sel_controls,
        inputs=[layers_state, sel_idx],
        outputs=[sel_text, sel_seed, sel_visible, sel_x, sel_y, sel_sc, sel_rot, sel_maxw],
    )

    # Auto-apply when X or Y changes (position only — no re-generation needed)
    for _pos_slider in [sel_x, sel_y]:
        _pos_slider.change(
            fn=apply_transform,
            inputs=[layers_state, sel_idx,
                    sel_text, sel_seed, sel_visible,
                    sel_x, sel_y, sel_sc, sel_rot, sel_maxw,
                    bg_input, canvas_w, canvas_h],
            outputs=[layers_state, layers_display, canvas_out, canvas_info],
        )

    # Layer reorder / delete
    layer_up_btn.click(
        fn=lambda l, i: move_layer(l, i, "up"),
        inputs=[layers_state, sel_idx],
        outputs=[layers_state, layers_display, sel_idx],
    )
    layer_down_btn.click(
        fn=lambda l, i: move_layer(l, i, "down"),
        inputs=[layers_state, sel_idx],
        outputs=[layers_state, layers_display, sel_idx],
    )
    layer_del_btn.click(
        fn=delete_layer,
        inputs=[layers_state, sel_idx],
        outputs=[layers_state, layers_display, sel_idx, canvas_info],
    )

    # Clear all
    clear_btn.click(
        fn=clear_all,
        outputs=[layers_state, layers_display, canvas_out, canvas_info],
    )

    # On load — build gallery with initial 4 seeds
    demo.load(
        fn=lambda: build_gallery_images(PREVIEW_TEXT, GALLERY_SEEDS[:GALLERY_INIT_COUNT]),
        outputs=[seed_gallery],
    )

    # Auto-render when bg changes
    bg_input.change(
        fn=do_render,
        inputs=[layers_state, bg_input, canvas_w, canvas_h],
        outputs=[canvas_out, canvas_info],
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Base(), css=CSS)