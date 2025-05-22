import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import math
import argparse
from math import pi

# Global default parameters
DEFAULT_PARAMS = {
    # Basic options
    'width': 400,
    'height': 320,
    'base_images': 1,
    
    # Font options
    'font_dir': '/home/ubuntu/sanskrit-ocr/datagen/fonts',
    'font': 'Sharad76-Regular.otf',
    
    # Generation-level augmentations
    'noise': 0.7,
    'aging': 0.6,
    'texture': 0.7,
    'stains': 0.6,
    'stain_intensity': 0.5,
    
    # Word-level options
    'word_position': 0.6,
    'ink_color': 0.5,
    'line_spacing': 0.4,
    'baseline': 0.3,
    'word_angle': 0.0,
    
    # Post-processing options
    'apply_transforms': True,
    'all_transforms': False,
    'rotation_max': 5.0,
    'brightness_var': 0.2,
    'contrast_var': 0.2,
    'noise_min': 0.01,
    'noise_max': 0.05,
    'blur_min': 0.5,
    'blur_max': 1.0
}

def _create_background(width, height, style, params):
    # Check if image directory is provided and use image background if available
    if 'image_dir' in params and os.path.exists(params['image_dir']):
        image_files = [f for f in os.listdir(params['image_dir']) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            img_path = os.path.join(params['image_dir'], random.choice(image_files))
            try:
                bg_img = Image.open(img_path).convert('RGB')
                # Resize the image to match required dimensions
                bg_img = bg_img.resize((width, height), Image.LANCZOS)
                return bg_img
            except Exception as e:
                print(f"Error loading background image {img_path}: {e}")
                # Fall back to synthetic backgrounds if image loading fails
    
    if style == "lined_paper":
        background = np.ones((height, width, 3), dtype=np.uint8) * [210, 180, 140]
        
        line_spacing = random.randint(15, 25)
        for y in range(0, height, line_spacing):
            line_width = random.randint(1, 2)
            darkness = random.randint(6, 20) * params["texture"]
            background[y:y+line_width, :, :] = np.clip(background[y:y+line_width, :, :] - darkness, 0, 255)
            
        noise = np.random.randint(0, int(15 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        stain_count = int(random.randint(2, 4) * params["stains"])
        for _ in range(stain_count):
            x = random.randint(0, width-100)
            y = random.randint(0, height-100)
            size = random.randint(20, 60)
            darkness = random.randint(8, 25) * params["stain_intensity"]
            shape = np.ones((size, size, 3), dtype=np.uint8) * darkness
            for i in range(size):
                for j in range(size):
                    dist = ((i - size/2)**2 + (j - size/2)**2) / (size/4)**2
                    if dist < 1:
                        alpha = (1 - dist) * random.uniform(0.4, 0.8) * params["stain_intensity"]
                        if y+i < height and x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] - shape[i, j, :] * alpha, 0, 255
                            )
    
    elif style == "old_paper":
        background = np.ones((height, width, 3), dtype=np.uint8) * [236, 222, 181]
        
        noise = np.random.randint(0, int(12 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        edge_width = width // 10
        for i in range(edge_width):
            factor = (edge_width - i) / edge_width * 15 * params["aging"]
            background[i, :, 2] = np.clip(background[i, :, 2] - factor, 0, 255)
            background[height-i-1, :, 2] = np.clip(background[height-i-1, :, 2] - factor, 0, 255)
            background[:, i, 2] = np.clip(background[:, i, 2] - factor, 0, 255)
            background[:, width-i-1, 2] = np.clip(background[:, width-i-1, 2] - factor, 0, 255)
    
    elif style == "birch":
        background = np.ones((height, width, 3), dtype=np.uint8) * [235, 225, 215]
        
        noise = np.random.randint(0, int(10 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        variation_count = int(150 * params["texture"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(10, 25)
            variation = random.randint(-6, 6) * params["texture"]
            
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
    
    else:  # "parchment"
        background = np.ones((height, width, 3), dtype=np.uint8) * [230, 215, 185]
        
        variation_count = int(400 * params["texture"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 12)
            variation = random.randint(-7, 7) * params["texture"]
            
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
        
        noise = np.random.randint(0, int(8 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(background)

def _render_sanskrit(text, font_path, output_path, width, height, font_size, style, ink_color, params):
    img = _create_background(width, height, style, params)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
        
        # Remove newlines and treat all text as one block
        words = text.strip().replace('\n', ' ').split()
        
        y_position = random.randint(25, 75)
        margin = 25  # Left and right margin
        
        # Available width for text
        available_width = width - 2 * margin
        space_width = draw.textlength(" ", font=font)
        
        current_line = []
        current_line_width = 0
        
        # Collect all lines first
        all_lines = []
        for word in words:
            word_width = draw.textlength(word, font=font)
            
            # Check if adding this word would exceed available width
            if current_line and current_line_width + space_width + word_width > available_width:
                all_lines.append(current_line)
                current_line = [word]
                current_line_width = word_width
            else:
                if current_line:
                    current_line_width += space_width + word_width
                else:
                    current_line_width = word_width
                current_line.append(word)
        
        # Add the last line if there's anything left
        if current_line:
            all_lines.append(current_line)
        
        # Render all lines
        for line in all_lines:
            # Center the line horizontally
            line_text = " ".join(line)
            line_width = draw.textlength(line_text, font=font)
            x_position = (width - line_width) // 2
            
            baseline_offset = random.randint(-2, 2) * params["baseline"]
            y_line_position = y_position + baseline_offset
            
            # Check if we've reached the bottom of the image
            if y_line_position + font_size > height - margin:
                break
            
            # Render each word in the line
            x_word_position = x_position
            for word in line:
                word_x_offset = int(random.uniform(-1.5, 1.5) * params["word_position"])
                word_y_offset = int(random.uniform(-1, 1) * params["word_position"])
                
                color_variation = int(random.randint(-3, 3) * params["ink_color"])
                word_color = (
                    np.clip(ink_color[0] + color_variation, 0, 255),
                    np.clip(ink_color[1] + color_variation, 0, 255),
                    np.clip(ink_color[2] + color_variation, 0, 255)
                )
                
                word_width = draw.textlength(word, font=font)
                word_height = font_size * 1.2
                
                if params["word_angle"] > 0:
                    # Apply rotation to individual word
                    word_angle = random.uniform(-2, 2) * params["word_angle"]
                    
                    diagonal = math.sqrt(word_width**2 + word_height**2)
                    padding = int(diagonal * 0.5)
                    
                    temp_width = int(diagonal + 2 * padding)
                    temp_height = int(diagonal + 2 * padding)
                    txt_img = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
                    txt_d = ImageDraw.Draw(txt_img)
                    
                    center_x = temp_width // 2 - word_width // 2
                    center_y = temp_height // 2 - word_height // 2
                    txt_d.text((center_x, center_y), word, font=font, fill=word_color + (255,))
                    
                    rotated = txt_img.rotate(word_angle, resample=Image.BICUBIC, expand=0, 
                                            center=(temp_width//2, temp_height//2))
                    
                    paste_x = int(x_word_position + word_x_offset - padding)
                    paste_y = int(y_line_position + word_y_offset - padding)
                    
                    img.paste(rotated, (paste_x, paste_y), rotated)
                else:
                    draw.text(
                        (x_word_position + word_x_offset, y_line_position + word_y_offset), 
                        word, fill=word_color, font=font
                    )
                
                x_word_position += word_width + space_width
            
            # Move to next line
            line_spacing_factor = 1.0 + (random.uniform(-0.1, 0.1) * params["line_spacing"])
            y_position += int(font_size * 1.2 * line_spacing_factor)
        
        if output_path is not None:
            img.save(output_path)
            print(f"Saved rendered Sanskrit to {output_path}")
        return img
        
    except Exception as e:
        print(f"Error rendering text with font {font_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ────────── Page warping helpers ──────────

def cylindrical_edge_warp(pil_img, side="left", strength=0.6, warp_portion=0.45):
    """
    Cylindrical bend on one side of the page.
    side         : "left" or "right"
    strength     : +ve bulges out, –ve bulges in. Magnitude ≈ tan(max_angle/2)
    warp_portion : fraction (0–1) of width from that edge to bend
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # width of the curved strip
    W = int(warp_portion * w)
    # fake focal length = radius of cylinder (pick something like the strip width)
    R = W / strength if strength != 0 else 1e9

    # Build meshgrid of pixel coords
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = X.astype(np.float32).copy()
    map_y = Y.astype(np.float32).copy()

    if side == "left":
        strip = X < W
        dx = W - X[strip]               # distance *into* the page from edge
    else:                               # right
        strip = X > (w - W)
        dx = X[strip] - (w - W)

    # angle on cylinder surface for those pixels
    theta = dx / R                      # radians
    # horizontal mapping (cylinder unrolled: x' = R*sinθ)
    displacement = R * np.sin(theta) - dx
    map_x[strip] += displacement

    # vertical scaling so text lines aren't stretched
    scale_y = np.cos(theta)
    map_y[strip] = (Y[strip] - h/2) / scale_y + h/2

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def washboard_warp(pil_img, amplitude=8, wavelength=120, phase=0.0,
                   decay_from_top=True):
    """Vertical sine ripples that run horizontally across the page."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # build a vector of vertical offsets – one per column
    x = np.arange(w, dtype=np.float32)
    dy = amplitude * np.sin(2*pi*x / wavelength + phase)

    if decay_from_top:
        atten = np.linspace(1, 0.2, h, dtype=np.float32)[:, None]  # fade as y increases
    else:
        atten = 1.0
    # broadcast to full map
    map_x, map_y = np.meshgrid(x, np.arange(h, dtype=np.float32))
    map_y += dy * atten

    warped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
# ────────────────────────────────────────

def _apply_postprocessing(original_image, output_dir, base_filename, params):
    all_images = [original_image]
    transforms = []
    
    def rotate_image(img, angle):
        bg_color = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)
        return rotated
    
    def adjust_brightness(img, factor):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def adjust_contrast(img, factor):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def add_noise(img, intensity):
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def blur_image(img, radius):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    transforms.append(("rotate", lambda img: rotate_image(img, 
                                            random.uniform(-params["rotation_max"], params["rotation_max"]))))
    transforms.append(("brightness", lambda img: adjust_brightness(img, 
                                            random.uniform(1.0-params["brightness_var"], 1.0+params["brightness_var"]))))
    transforms.append(("contrast", lambda img: adjust_contrast(img, 
                                            random.uniform(1.0-params["contrast_var"], 1.0+params["contrast_var"]))))
    transforms.append(("noise", lambda img: add_noise(img, 
                                            random.uniform(params["noise_min"], params["noise_max"]))))
    transforms.append(("blur", lambda img: blur_image(img, 
                                            random.uniform(params["blur_min"], params["blur_max"]))))
    
    # Add new page warping transformations
    transforms.append(("washboard", lambda img: washboard_warp(
        img,
        amplitude=random.uniform(6, 12),
        wavelength=random.uniform(90, 150),
        phase=random.uniform(0, 2*pi),
        decay_from_top=random.choice([True, False]))))
        
    transforms.append(("cylinder", lambda img: cylindrical_edge_warp(
        img,
        side=random.choice(["left", "right"]),
        strength=random.uniform(0.4, 0.8) * random.choice([1, -1]),
        warp_portion=random.uniform(0.35, 0.5))))
    
    if params["all_transforms"]:
        selected_transforms = transforms
    else:
        n_transforms = random.randint(1, min(3, len(transforms)))
        selected_transforms = random.sample(transforms, n_transforms)
    
    for transform_name, transform_func in selected_transforms:
        transformed_img = transform_func(original_image)
        transformed_filename = f"{base_filename}_{transform_name}.png"
        transformed_path = os.path.join(output_dir, transformed_filename)
        transformed_img.save(transformed_path)
        print(f"Saved transformed image to {transformed_path}")
        all_images.append(transformed_img)
    
    if len(selected_transforms) > 1:
        combined_img = original_image.copy()
        for _, transform_func in selected_transforms:
            combined_img = transform_func(combined_img)
        
        combined_filename = f"{base_filename}_combined.png"
        combined_path = os.path.join(output_dir, combined_filename)
        combined_img.save(combined_path)
        print(f"Saved combined transformation to {combined_path}")
        all_images.append(combined_img)
    
    return all_images

def generate_sanskrit_samples(text, font_path=None, output_dir=None, params=None):
    # Use default params if none provided, updating with any provided values
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        # Create a copy of default params and update with provided values
        params = {**DEFAULT_PARAMS, **params}
    
    # Set default font path if not provided
    if font_path is None:
        font_path = os.path.join(params['font_dir'], params['font'])
    
    if not os.path.exists(font_path):
        print(f"Error: Font not found at {font_path}")
        return [] if output_dir is None else None
    
    styles = ["lined_paper", "old_paper", "birch", "parchment"]
    
    ink_colors = {
        "lined_paper": (60, 30, 10),
        "old_paper": (20, 20, 20),
        "birch": (50, 20, 10),
        "parchment": (10, 10, 10)
    }
    
    width, height = params['width'], params['height']
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Randomly sample styles for the total number of base images
    sampled_styles = random.choices(styles, k=params['base_images'])
    style_counts = {style: sampled_styles.count(style) for style in styles}
    print(f"Randomly selected styles: {style_counts}")
    
    base_images = []
    image_counter = 0
    
    # Generate randomly sampled base images
    for style, count in style_counts.items():
        for i in range(count):
            image_counter += 1
            
            # Randomly select a font size between 12 and 18
            font_size = random.randint(12, 18)
            print(f"Using font size {font_size} for {style}_{i+1}")
            
            # If output_dir is provided, save to file, otherwise just render
            output_path = os.path.join(output_dir, f"sanskrit_{style}_{i+1}.png") if output_dir else None
            
            img = _render_sanskrit(
                text=text,
                font_path=font_path,
                output_path=output_path,
                width=width,
                height=height,
                font_size=font_size,
                style=style,
                ink_color=ink_colors[style],
                params=params
            )
            
            if img:
                base_images.append(img)
                
                if params['apply_transforms'] and output_dir:
                    base_filename = f"sanskrit_{style}_{i+1}"
                    transformed_images = _apply_postprocessing(img, output_dir, base_filename, params)
                    base_images.extend(transformed_images)
    
    return base_images if output_dir is None else None

def main():
    sanskrit_text = """ज्ञानं परमं ध्येयम्। ज्ञानात् सत्यं प्रकाशते। सत्येन मुक्तिः प्राप्यते। मुक्तिः परमं सुखम्। तस्मात् ज्ञानं समभ्यसेत्। विद्या ददाति विनयम्। विनयात् याति पात्रताम्। पात्रत्वात् धनमाप्नोति। धनात् धर्मं ततः सुखम्॥"""
    
    parser = argparse.ArgumentParser(description='Generate Sanskrit text samples with word-level and image augmentations')
    
    # Basic options
    basic = parser.add_argument_group('Basic Options')
    basic.add_argument('--output-dir', type=str, default='data/synthetic/images',
                      help='Output directory for generated images')
    basic.add_argument('--width', type=int, default=DEFAULT_PARAMS['width'],
                      help='Width of output images')
    basic.add_argument('--height', type=int, default=DEFAULT_PARAMS['height'], 
                      help='Height of output images')
    basic.add_argument('--base-images', type=int, default=DEFAULT_PARAMS['base_images'],
                      help='Total number of base images to generate (randomly sampled styles)')
    
    # Font options
    font = parser.add_argument_group('Font Options')
    font.add_argument('--font-dir', type=str, default=DEFAULT_PARAMS['font_dir'],
                    help='Directory containing font files')
    font.add_argument('--font', type=str, default=DEFAULT_PARAMS['font'],
                    help='Font filename within the font directory')
    
    # Generation-level augmentations
    gen = parser.add_argument_group('Generation-Level Augmentations')
    gen.add_argument('--noise', type=float, default=DEFAULT_PARAMS['noise'],
                   help='Background noise intensity (0.0-1.0)')
    gen.add_argument('--aging', type=float, default=DEFAULT_PARAMS['aging'],
                   help='Edge aging effect (0.0-1.0)')
    gen.add_argument('--texture', type=float, default=DEFAULT_PARAMS['texture'],
                   help='Texture variation (0.0-1.0)')
    gen.add_argument('--stains', type=float, default=DEFAULT_PARAMS['stains'],
                   help='Number of stains (0.0-1.0)')
    gen.add_argument('--stain-intensity', type=float, default=DEFAULT_PARAMS['stain_intensity'],
                   help='Intensity of stain effects (0.0-1.0)')
    gen.add_argument('--image-dir', type=str, default='',
                    help='Directory to randomly sample background image from. If left empty, no image backgrounds will be used.')
    
    # Word-level options
    gen.add_argument('--word-position', type=float, default=DEFAULT_PARAMS['word_position'],
                   help='Random word position variation (0.0-1.0)')
    gen.add_argument('--ink-color', type=float, default=DEFAULT_PARAMS['ink_color'],
                   help='Ink color variation (0.0-1.0)')
    gen.add_argument('--line-spacing', type=float, default=DEFAULT_PARAMS['line_spacing'],
                   help='Random line spacing (0.0-1.0)')
    gen.add_argument('--baseline', type=float, default=DEFAULT_PARAMS['baseline'],
                   help='Baseline wobble effect (0.0-1.0)')
    gen.add_argument('--word-angle', type=float, default=DEFAULT_PARAMS['word_angle'],
                   help='Random word angle (0.0-1.0)')
    
    # Post-processing options
    post = parser.add_argument_group('Post-Processing Augmentations')
    post.add_argument('--no-transforms', dest='apply_transforms', action='store_false',
                    help='Disable post-processing transforms')
    post.add_argument('--all-transforms', action='store_true',
                    help='Apply all transforms instead of random subset')
    post.add_argument('--rotation-max', type=float, default=DEFAULT_PARAMS['rotation_max'],
                    help='Maximum rotation angle in degrees')
    post.add_argument('--brightness-var', type=float, default=DEFAULT_PARAMS['brightness_var'],
                    help='Brightness variation factor (0.0-1.0)')
    post.add_argument('--contrast-var', type=float, default=DEFAULT_PARAMS['contrast_var'],
                    help='Contrast variation factor (0.0-1.0)')
    post.add_argument('--noise-min', type=float, default=DEFAULT_PARAMS['noise_min'],
                    help='Minimum noise intensity for transforms')
    post.add_argument('--noise-max', type=float, default=DEFAULT_PARAMS['noise_max'],
                    help='Maximum noise intensity for transforms')
    post.add_argument('--blur-min', type=float, default=DEFAULT_PARAMS['blur_min'],
                    help='Minimum blur radius')
    post.add_argument('--blur-max', type=float, default=DEFAULT_PARAMS['blur_max'],
                    help='Maximum blur radius')
    
    parser.set_defaults(apply_transforms=DEFAULT_PARAMS['apply_transforms'], 
                       all_transforms=DEFAULT_PARAMS['all_transforms'])
    args = parser.parse_args()
    
    # Convert args to dict, excluding output_dir
    params = {k: v for k, v in vars(args).items() if k != 'output_dir'}
    
    generate_sanskrit_samples(
        text=sanskrit_text,
        font_path=os.path.join(params['font_dir'], params['font']),
        output_dir=args.output_dir,
        params=params
    )

if __name__ == "__main__":
    main()

    # with open('/Users/rohan/Desktop/indic_merged.txt', 'r') as text:
    #     lines = []
    #     for i in range(10):
    #         line = text.readline()[:-1]
    #         lines.append(line)

    #     text = ' ।\n'.join(lines) + '।।'
    #     print(text)
    #     images = generate_sanskrit_samples(text)
    #     print(len(images))
    #     images[0].show()

    #     import time; time.sleep(20)