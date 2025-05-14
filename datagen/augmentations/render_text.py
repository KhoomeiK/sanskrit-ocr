#!/usr/bin/env python3
"""
Render a Sanskrit sample with correct Devanāgarī conjuncts for every
TrueType/OpenType font under datagen/fonts/** and build a preview grid.

Core change:
    *  uses Pillow’s RAQM layout engine instead of manual glyph painting
      so ligatures such as क्ष, ज्ञ, स्त्र, etc. appear correctly.
"""

import os
import glob
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features
from fontTools.ttLib import TTFont

# ============================================================================
# --- Configuration ----------------------------------------------------------
# ============================================================================

SANSKRIT_TEXT = (
    "ज्ञानं परमं ध्येयम्। ज्ञानात् सत्यं प्रकाशते।\n"
    "सत्येन मुक्तिः प्राप्यते। मुक्तिः परमं सुखम्।\n"
    "तस्मात् ज्ञानं समभ्यसेत्। विद्या ददाति विनयम्।\n"
    "विनयात् याति पात्रताम्। पात्रत्वात् धनमाप्नोति।\n"
    "धनात् धर्मं ततः सुखम्॥"
)

EXCLUDE_FONTS = {
    "bssym1.ttf", "bssym2.ttf", "bssym3.ttf", "bssym4.ttf", "bssym5.ttf",
    "Devanagari New Normal.ttf", "agadhns_.ttf", "hindi-5.ttf",
}

# ——— Rendering defaults
FONT_SIZE       = 28
SAMPLE_WIDTH    = 1000
SAMPLE_HEIGHT   = 350
GRID_COLUMNS    = 5
BG_COLOR        = "white"
FG_COLOR        = "black"

# ============================================================================
# --- Helper functions -------------------------------------------------------
# ============================================================================

def check_raqm_available() -> None:
    """
    Abort early if Pillow was compiled without RAQM; complex‑script shaping
    will not work otherwise.
    """
    if not features.check_feature("raqm"):           # ⬅ Pillow⇢RAQM hook
        raise RuntimeError(
            "Pillow on this system was built without RAQM "
            "(FreeType + HarfBuzz + FriBiDi). "
            "Reinstall Pillow with RAQM enabled to render Devanāgarī correctly."
        )

def find_all_fonts(base_dir: str = "datagen/fonts") -> list[str]:
    """
    Traverse *base_dir* recursively and return every .ttf / .otf file that
    is **not** on the user’s exclusion list.
    """
    font_paths: list[str] = []
    for ext in (".ttf", ".otf"):
        font_paths.extend(Path(base_dir).rglob(f"*{ext}"))

    return [
        str(p) for p in font_paths
        if p.name not in EXCLUDE_FONTS
    ]

def devanagari_support(font_path: str) -> dict:
    """
    Return basic coverage information for the Devanāgarī block (U+0900–097F).
    """
    try:
        tt = TTFont(font_path)
        cmap = tt.getBestCmap()
        devanagari_range = range(0x0900, 0x0980)
        covered = sum(1 for cp in devanagari_range if cp in cmap)

        return {
            "supports_devanagari": covered > 0,
            "coverage_pct":       covered / len(devanagari_range) * 100,
        }
    except Exception as exc:
        return {"supports_devanagari": False, "coverage_pct": 0, "error": str(exc)}

def can_render_sample(font_path: str, sample: str = SANSKRIT_TEXT) -> bool:
    """
    Fast static test: every non‑ASCII code‑point in *sample* must exist in the
    font’s cmap.  (Actual shaping is handled later by RAQM.)
    """
    try:
        cmap = TTFont(font_path).getBestCmap()
        return all((ord(c) in cmap) or (ord(c) < 128) for c in sample)
    except Exception:
        return False

def render_sample(font_path: str,
                  text: str = SANSKRIT_TEXT,
                  font_size: int = FONT_SIZE,
                  img_w: int = SAMPLE_WIDTH,
                  img_h: int = SAMPLE_HEIGHT) -> Image.Image:
    """
    Render *text* with *font_path* using RAQM so conjuncts appear correctly.
    """
    # 1 · create blank canvas
    img  = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 2 · load font with RAQM layout engine
    font = ImageFont.truetype(
        font_path,
        font_size,
        layout_engine=ImageFont.Layout.RAQM,      # ✨ complex‑text shaping
    )

    # 3 · draw file name (always ascii) with default font
    draw.text((10, 8), os.path.basename(font_path),
              font=ImageFont.load_default(), fill=FG_COLOR)

    # 4 · baseline y‑offset so the sample starts below the file name
    y_start = 40
    text_box = (20, y_start, img_w - 20, img_h - 20)

    # RAQM‑powered multiline render; language tag helps hint selection
    draw.multiline_text(
        (text_box[0], text_box[1]),
        text,
        font=font,
        fill=FG_COLOR,
        spacing=6,
        align="left",
        language="sa",          # ISO 639‑1 for Sanskrit
        direction="ltr",
    )

    # optional visual baseline
    draw.line([(10, img_h - 100), (img_w - 10, img_h - 100)],
              fill=(200, 200, 200), width=2)

    return img

# ============================================================================
# --- Main grid builder ------------------------------------------------------
# ============================================================================

def create_font_grid(output_path: str = "sanskrit_fonts_grid.png",
                     columns: int = GRID_COLUMNS) -> None:
    """
    Collect all usable fonts, render a sample with each, and assemble them into
    a single preview grid + individual PNGs.
    """
    check_raqm_available()

    font_files = find_all_fonts()
    print(f"→ Scanned {len(font_files)} candidate font files")

    usable: list[dict] = []
    for path in font_files:
        meta = devanagari_support(path)
        if not meta.get("supports_devanagari"):
            continue
        if not can_render_sample(path):
            continue
        usable.append({"path": path, "coverage": meta["coverage_pct"]})

    if not usable:
        raise RuntimeError("No fonts found that can render the sample text.")

    # Sort by coverage (descending) then filename for tiebreak
    usable.sort(key=lambda d: (-d["coverage"], os.path.basename(d["path"])))
    print(f"→ {len(usable)} fonts passed coverage and cmap checks")

    # Compute grid size
    rows = math.ceil(len(usable) / columns)
    grid_w = columns * (SAMPLE_WIDTH + 10) + 10
    grid_h = rows    * (SAMPLE_HEIGHT + 10) + 10
    grid   = Image.new("RGB", (grid_w, grid_h), BG_COLOR)

    # Render each sample & paste into grid
    for idx, info in enumerate(usable):
        row, col = divmod(idx, columns)
        x = col * (SAMPLE_WIDTH  + 10) + 10
        y = row * (SAMPLE_HEIGHT + 10) + 10
        sample_img = render_sample(info["path"])
        grid.paste(sample_img, (x, y))

    # --- output -----------------------------------------------------------------
    grid.save(output_path, optimize=True, quality=95)
    print(f"✔️  Grid saved → {output_path}")

    # per‑font PNGs (optional; keeps your original behaviour)
    out_dir = Path("font_samples")
    out_dir.mkdir(exist_ok=True)

    for info in usable:
        fname = out_dir / f"{Path(info['path']).name}.png"
        render_sample(info["path"]).save(fname, optimize=True, quality=95)

    print(f"✔️  Individual samples saved → {out_dir}/")

# ============================================================================
# --- Entry‑point ------------------------------------------------------------
# ============================================================================

if __name__ == "__main__":
    create_font_grid(font_size=24, columns=GRID_COLUMNS)
