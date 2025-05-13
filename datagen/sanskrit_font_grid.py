#!/usr/bin/env python3
import os
import glob
import math
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from collections import defaultdict

# Sanskrit text
SANSKRIT_TEXT = """ज्ञानं परमं ध्येयम्। ज्ञानात् सत्यं प्रकाशते।
सत्येन मुक्तिः प्राप्यते। मुक्तिः परमं सुखम्।
तस्मात् ज्ञानं समभ्यसेत्। विद्या ददाति विनयम्।
विनयात् याति पात्रताम्। पात्रत्वात् धनमाप्नोति।
धनात् धर्मं ततः सुखम्॥"""

# Fonts to exclude
EXCLUDE_FONTS = [
    "bssym1.ttf", "bssym2.ttf", "bssym3.ttf", "bssym4.ttf", "bssym5.ttf",
    "Devanagari New Normal.ttf", "agadhns_.ttf", "hindi-5.ttf"
]

def find_all_fonts(base_dir="datagen/fonts"):
    font_extensions = ['.ttf', '.otf']
    font_files = []
    
    for ext in font_extensions:
        font_files.extend(glob.glob(f"{base_dir}/**/*{ext}", recursive=True))
    
    # Filter out excluded fonts
    filtered_fonts = []
    for font_path in font_files:
        filename = os.path.basename(font_path)
        if filename not in EXCLUDE_FONTS:
            filtered_fonts.append(font_path)
    
    return filtered_fonts

def check_devanagari_support(font_path):
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        devanagari_range = range(0x0900, 0x097F + 1)
        supported_chars = [hex(c) for c in devanagari_range if c in cmap]
        support_percentage = len(supported_chars) / len(devanagari_range) * 100
        
        return {
            "supports_devanagari": len(supported_chars) > 0,
            "coverage_percentage": support_percentage,
            "supported_chars": len(supported_chars),
            "total_devanagari_chars": len(devanagari_range)
        }
    except Exception as e:
        return {
            "error": str(e),
            "supports_devanagari": False,
            "coverage_percentage": 0
        }

def can_render_sanskrit(font_path, text):
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        
        # Get Unicode code points in the sample text
        text_unicode = [ord(c) for c in text]
        
        # Check if all characters are supported
        for code_point in text_unicode:
            if code_point > 127 and code_point not in cmap:  # Skip ASCII
                return False
        
        return True
    except Exception:
        return False

def render_sanskrit_with_font(font_path, text, font_size=28, width=1000, height=350):
    try:
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        font = ImageFont.truetype(font_path, font_size)
        
        # Draw filename
        filename = os.path.basename(font_path)
        draw.text((10, 10), filename, font=ImageFont.truetype("Arial.ttf", 16), fill='black')
        
        # Draw Sanskrit text
        lines = text.split('\n')
        y_pos = 40
        for i, line in enumerate(lines):
            line_spacing = font_size + 20 if i == len(lines) - 1 else font_size + 15
            draw.text((20, y_pos), line, font=font, fill='black')
            y_pos += line_spacing
        
        # Baseline indicator
        draw.line([(10, height-100), (width-10, height-100)], fill=(200, 200, 200), width=2)
        
        return img
    except Exception as e:
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Error with {os.path.basename(font_path)}: {str(e)}", fill='red')
        return img

def create_font_grid(output_path="sanskrit_fonts_grid.png", font_size=28, columns=2):
    font_files = find_all_fonts()
    print(f"Found {len(font_files)} font files")
    
    fonts_data = []
    
    for font_path in font_files:
        result = check_devanagari_support(font_path)
        
        if "error" in result and result["error"]:
            print(f"Error analyzing {font_path}: {result['error']}")
            continue
        
        if not result["supports_devanagari"]:
            print(f"Font {font_path} does not support Devanagari, skipping")
            continue
        
        if not can_render_sanskrit(font_path, SANSKRIT_TEXT):
            print(f"Font {font_path} cannot render all characters in the Sanskrit text, skipping")
            continue
        
        fonts_data.append({
            "path": font_path,
            "coverage": result["coverage_percentage"],
            "filename": os.path.basename(font_path)
        })
    
    # Sort by coverage
    sorted_fonts = sorted(fonts_data, key=lambda x: (-x["coverage"], x["filename"]))
    
    print(f"Creating grid with {len(sorted_fonts)} Sanskrit fonts")
    
    font_width = 1000
    font_height = 350
    padding = 10
    
    cols = columns
    rows = math.ceil(len(sorted_fonts) / cols)
    
    grid_width = cols * (font_width + padding) + padding
    grid_height = rows * (font_height + padding) + padding
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    for i, font_data in enumerate(sorted_fonts):
        row = i // cols
        col = i % cols
        
        x = col * (font_width + padding) + padding
        y = row * (font_height + padding) + padding
        
        font_img = render_sanskrit_with_font(
            font_data["path"], 
            SANSKRIT_TEXT, 
            font_size=font_size,
            width=font_width,
            height=font_height
        )
        
        grid_img.paste(font_img, (x, y))
    
    grid_img.save(output_path, quality=95, optimize=True)
    print(f"Grid saved to {output_path}")
    
    # Create individual font samples
    output_dir = "font_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for font_data in sorted_fonts:
        font_img = render_sanskrit_with_font(
            font_data["path"], 
            SANSKRIT_TEXT, 
            font_size=font_size,
            width=font_width,
            height=font_height
        )
        
        filename = os.path.basename(font_data["path"])
        font_img.save(f"{output_dir}/{filename}.png", quality=95, optimize=True)
    
    print(f"Individual font samples saved in {output_dir}/")

if __name__ == "__main__":
    create_font_grid(font_size=24, columns=5) 