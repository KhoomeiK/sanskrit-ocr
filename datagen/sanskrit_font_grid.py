from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, features
from fontTools.ttLib import TTFont
import os
import math

def check_raqm_available():
    if not features.check_feature("raqm"):
        raise RuntimeError(
            "Pillow was built without RAQM support. "
            "Reinstall Pillow with RAQM enabled to shape Devanāgarī correctly."
        )

def find_all_fonts(base_dir, exclude_fonts):
    paths = []
    for ext in (".ttf", ".otf"):
        paths.extend(Path(base_dir).rglob(f"*{ext}"))
    return [str(p) for p in paths if p.name not in exclude_fonts]

def devanagari_support(font_path):
    try:
        tt = TTFont(font_path)
        cmap = tt.getBestCmap()
        total = 0x0980 - 0x0900
        got = sum(1 for cp in range(0x0900, 0x0980) if cp in cmap)
        return {"supports_devanagari": got > 0, "coverage_pct": got / total * 100}
    except Exception:
        return {"supports_devanagari": False, "coverage_pct": 0}

def can_render_sample(font_path, sample):
    try:
        cmap = TTFont(font_path).getBestCmap()
        return all((ord(c) < 128) or (ord(c) in cmap) for c in sample)
    except Exception:
        return False

def render_sample(font_path, text, font_size, img_w, img_h, name_font_size, bg_color, fg_color):
    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        font_path,
        font_size,
        layout_engine=ImageFont.Layout.RAQM
    )

    try:
        name_font = ImageFont.truetype("DejaVuSans.ttf", name_font_size)
    except IOError:
        name_font = ImageFont.load_default()

    draw.text((10, 8), os.path.basename(font_path),
              font=name_font, fill=fg_color)

    draw.multiline_text(
        (20, 40),
        text,
        font=font,
        fill=fg_color,
        spacing=6,
        align="left",
        language="sa",
        direction="ltr"
    )

    return img

def create_font_grid(output_path, font_size, columns, sample_width, sample_height, 
                   spacing, bg_color, fg_color, sanskrit_text, name_font_size, base_dir, exclude_fonts):
    check_raqm_available()

    fonts = find_all_fonts(base_dir, exclude_fonts)
    print(f"→ Found {len(fonts)} font files")

    usable = []
    for path in fonts:
        meta = devanagari_support(path)
        if not meta["supports_devanagari"]:
            continue
        if not can_render_sample(path, sanskrit_text):
            continue
        usable.append((path, meta["coverage_pct"]))

    if not usable:
        raise RuntimeError("No fonts found that can render the sample text.")

    usable.sort(key=lambda x: (-x[1], os.path.basename(x[0])))
    print(f"→ {len(usable)} fonts passed coverage & cmap checks")

    rows = math.ceil(len(usable) / columns)
    gw = columns * (sample_width + spacing) - spacing
    gh = rows * (sample_height + spacing) - spacing
    grid = Image.new("RGB", (gw, gh), bg_color)

    for idx, (path, _) in enumerate(usable):
        r, c = divmod(idx, columns)
        x = c * (sample_width + spacing)
        y = r * (sample_height + spacing)
        sample_img = render_sample(path, sanskrit_text, font_size, sample_width, sample_height, 
                                  name_font_size, bg_color, fg_color)
        grid.paste(sample_img, (x, y))

    grid.save(output_path, optimize=True, quality=95)
    print(f"✔ Grid saved to {output_path}")

if __name__ == "__main__":
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
    
    DEFAULT_FONT_SIZE = 24
    NAME_FONT_SIZE = 20
    SAMPLE_WIDTH = 500
    SAMPLE_HEIGHT = 250
    DEFAULT_COLUMNS = 5
    SPACING = 2
    BG_COLOR = "white"
    FG_COLOR = "black"
    BASE_DIR = "datagen/fonts"
    OUTPUT_PATH = "datagen/sanskrit_fonts_grid.png"
    
    create_font_grid(
        OUTPUT_PATH, 
        DEFAULT_FONT_SIZE, 
        DEFAULT_COLUMNS, 
        SAMPLE_WIDTH, 
        SAMPLE_HEIGHT, 
        SPACING, 
        BG_COLOR, 
        FG_COLOR, 
        SANSKRIT_TEXT, 
        NAME_FONT_SIZE, 
        BASE_DIR, 
        EXCLUDE_FONTS
    )