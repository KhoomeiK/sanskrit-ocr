import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import math
import argparse

def _add_enhanced_degradation(background, width, height, params):
    """Add physically accurate degradation patterns to manuscripts."""
    # Convert to numpy array for pixel manipulation
    bg_array = np.array(background)
    
    # 1. INSECT DAMAGE IMPLEMENTATION
    if hasattr(params, 'insect_damage') and params.insect_damage > 0:
        # Create 1-3 clustered damage paths (insects typically create trails)
        cluster_count = int(random.randint(1, 3) * params.insect_damage)
        
        for _ in range(cluster_count):
            # Determine damage path characteristics
            edge_start = random.choice(['top', 'bottom', 'left', 'right'])
            if edge_start == 'top':
                start_x = random.randint(width//4, 3*width//4)
                start_y = random.randint(0, height//10)
            elif edge_start == 'bottom':
                start_x = random.randint(width//4, 3*width//4)
                start_y = random.randint(9*height//10, height-1)
            elif edge_start == 'left':
                start_x = random.randint(0, width//10)
                start_y = random.randint(height//4, 3*height//4)
            else:  # right
                start_x = random.randint(9*width//10, width-1)
                start_y = random.randint(height//4, 3*height//4)
            
            # Create path for insect damage
            path_length = random.randint(5, 15)
            current_x, current_y = start_x, start_y
            path_points = [(current_x, current_y)]
            
            for i in range(path_length):
                angle = random.uniform(0, 2*np.pi)
                distance = random.randint(10, 30)
                current_x += int(np.cos(angle) * distance)
                current_y += int(np.sin(angle) * distance)
                
                # Keep within bounds
                current_x = max(0, min(width-1, current_x))
                current_y = max(0, min(height-1, current_y))
                
                path_points.append((current_x, current_y))
            
            # Create holes along the path
            for path_idx in range(len(path_points)-1):
                x1, y1 = path_points[path_idx]
                x2, y2 = path_points[path_idx+1]
                
                steps = max(abs(x2-x1), abs(y2-y1))
                if steps == 0:
                    continue
                
                for step in range(steps):
                    x = x1 + (x2-x1) * step // steps
                    y = y1 + (y2-y1) * step // steps
                    
                    if random.random() > 0.7:
                        continue
                    
                    hole_size = random.randint(1, 3)
                    if random.random() < 0.1:
                        hole_size = random.randint(3, 5)
                    
                    for i in range(-hole_size, hole_size+1):
                        for j in range(-hole_size, hole_size+1):
                            dist = np.sqrt(i**2 + j**2)
                            
                            if dist > hole_size:
                                continue
                                
                            if 0 <= y+i < height and 0 <= x+j < width:
                                if dist < hole_size * 0.7:
                                    bg_array[y+i, x+j] = [255, 255, 255]
                                else:
                                    darkness = int(40 * (1 - (dist / hole_size)))
                                    bg_array[y+i, x+j] = np.clip(bg_array[y+i, x+j] - darkness, 0, 255)
    
    # Add water damage implementation here (abbreviated for this step)
    if hasattr(params, 'water_damage') and params.water_damage > 0:
        # Basic water damage implementation
        # Will be expanded in later steps
        pass
    
    # Add edge deterioration implementation here (abbreviated for this step)
    if hasattr(params, 'edge_deterioration') and params.edge_deterioration > 0:
        # Basic edge deterioration implementation
        # Will be expanded in later steps
        pass
        
    return Image.fromarray(bg_array)

def _enhance_text_layout(draw, text, font, x_position, y_position, line_index, width, font_size, params):
    """Create more natural handwritten text appearance with advanced baseline and spacing variations."""
    # Calculate baseline wobble
    if params.baseline > 0:
        # Primary wave (slow undulation across line)
        frequency1 = random.uniform(0.5, 1.5)
        amplitude1 = random.uniform(0.5, 1.0) * params.baseline
        phase1 = random.uniform(0, 2*np.pi)
        
        # Secondary wave (faster variations)
        frequency2 = random.uniform(3.0, 6.0)
        amplitude2 = random.uniform(0.1, 0.3) * params.baseline
        phase2 = random.uniform(0, 2*np.pi)
        
        # Calculate baseline offset
        normalized_x = x_position / width
        wave1 = amplitude1 * np.sin(frequency1 * 2*np.pi * normalized_x + phase1)
        wave2 = amplitude2 * np.sin(frequency2 * 2*np.pi * normalized_x + phase2)
        
        baseline_offset = int(wave1 + wave2)
    else:
        baseline_offset = 0
    
    # Calculate character spacing
    word_x_offset = int(random.uniform(-2, 2) * params.word_position)
    word_y_offset = int(random.uniform(-1.5, 1.5) * params.word_position)
    
    # Create positions for individual characters
    char_positions = []
    x_char = x_position + word_x_offset
    
    for i, char in enumerate(text):
        char_width = draw.textlength(char, font=font)
        
        # Position calculation
        pos_x = x_char
        pos_y = y_position + baseline_offset + word_y_offset
        
        # Character variation
        char_x_var = random.uniform(-0.3, 0.3) * params.word_position * 0.3
        char_y_var = random.uniform(-0.2, 0.2) * params.word_position * 0.3
        
        # Apply variations with constraints
        final_x = pos_x + char_x_var
        final_y = pos_y + char_y_var
        
        # Constrain positions
        x_min = x_char - char_width * 0.1
        x_max = x_char + char_width * 1.1
        final_x = max(x_min, min(final_x, x_max))
        
        y_min = y_position - font_size * 0.2
        y_max = y_position + font_size * 0.2
        final_y = max(y_min, min(final_y, y_max))
        
        # Store position
        char_positions.append((final_x, final_y))
        
        # Advance position
        x_char += char_width
    
    return char_positions


def _create_background(width, height, style, params):
    if style == "lined_paper":
        background = np.ones((height, width, 3), dtype=np.uint8) * [210, 180, 140]
        
        line_spacing = random.randint(15, 25)
        for y in range(0, height, line_spacing):
            line_width = random.randint(1, 2)
            darkness = random.randint(6, 20) * params.texture
            background[y:y+line_width, :, :] = np.clip(background[y:y+line_width, :, :] - darkness, 0, 255)
            
        noise = np.random.randint(0, int(15 * params.noise), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        stain_count = int(random.randint(2, 4) * params.stains)
        for _ in range(stain_count):
            x = random.randint(0, width-100)
            y = random.randint(0, height-100)
            size = random.randint(20, 60)
            darkness = random.randint(8, 25) * params.stain_intensity
            shape = np.ones((size, size, 3), dtype=np.uint8) * darkness
            for i in range(size):
                for j in range(size):
                    dist = ((i - size/2)**2 + (j - size/2)**2) / (size/4)**2
                    if dist < 1:
                        alpha = (1 - dist) * random.uniform(0.4, 0.8) * params.stain_intensity
                        if y+i < height and x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] - shape[i, j, :] * alpha, 0, 255
                            )
    
    elif style == "old_paper":
        background = np.ones((height, width, 3), dtype=np.uint8) * [236, 222, 181]
        
        noise = np.random.randint(0, int(12 * params.noise), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        edge_width = width // 10
        for i in range(edge_width):
            factor = (edge_width - i) / edge_width * 15 * params.aging
            background[i, :, 2] = np.clip(background[i, :, 2] - factor, 0, 255)
            background[height-i-1, :, 2] = np.clip(background[height-i-1, :, 2] - factor, 0, 255)
            background[:, i, 2] = np.clip(background[:, i, 2] - factor, 0, 255)
            background[:, width-i-1, 2] = np.clip(background[:, width-i-1, 2] - factor, 0, 255)
    
    elif style == "birch":
        background = np.ones((height, width, 3), dtype=np.uint8) * [235, 225, 215]
        
        noise = np.random.randint(0, int(10 * params.noise), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        variation_count = int(150 * params.texture)
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(10, 25)
            variation = random.randint(-6, 6) * params.texture
            
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
    
    else:  # "parchment"
        background = np.ones((height, width, 3), dtype=np.uint8) * [230, 215, 185]
        
        variation_count = int(400 * params.texture)
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 12)
            variation = random.randint(-7, 7) * params.texture
            
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
        
        noise = np.random.randint(0, int(8 * params.noise), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
    if hasattr(params, 'vertical_lines') and hasattr(params, 'horizontal') and params.horizontal and params.vertical_lines:
        # Left and right margin (typically 5-8% of width)
        margin = int(width * 0.06)
        line_width = random.randint(1, 2)
        line_color = [60, 60, 60]  # Dark gray
        
        # Left vertical line
        background[:, margin:margin+line_width, :] = line_color
        
        # Right vertical line
        background[:, width-margin:width-margin+line_width, :] = line_color
    
    result_img = Image.fromarray(background)
    
    # Apply enhanced degradation if parameters are available
    if (hasattr(params, 'insect_damage') and params.insect_damage > 0 or
        hasattr(params, 'water_damage') and params.water_damage > 0 or
        hasattr(params, 'edge_deterioration') and params.edge_deterioration > 0):
        result_img = _add_enhanced_degradation(result_img, width, height, params)

    return result_img

def _render_sanskrit(text, font_path, output_path, width, height, font_size, style, ink_color, params):
    img = _create_background(width, height, style, params)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
        
        # Remove newlines and treat all text as one block
        words = text.strip().replace('\n', ' ').split()
        
        # Initial position - adjusted for layout type
        if hasattr(params, 'horizontal') and params.horizontal:
            # Start text higher in the image for horizontal layout
            y_position = random.randint(height // 8, height // 4)
            
            # Set wider margins to account for vertical ruling lines
            left_margin = int(width * 0.08)
            right_margin = int(width * 0.08)
            
            # Available width for text
            available_width = width - left_margin - right_margin
        else:
            # Traditional vertical manuscript layout (from original code)
            y_position = random.randint(25, 75)
            margin = 25  # Left and right margin
            left_margin = margin
            right_margin = margin
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
        for line_index, line in enumerate(all_lines):
            # Center the line horizontally
            line_text = " ".join(line)
            line_width = draw.textlength(line_text, font=font)
            
            # Adjust horizontal position based on layout
            if hasattr(params, 'horizontal') and params.horizontal:
                x_position = left_margin + (available_width - line_width) // 2
            else:
                x_position = (width - line_width) // 2
            
            baseline_offset = random.randint(-2, 2) * params.baseline
            y_line_position = y_position + baseline_offset
            
            # Adjust bottom margin check based on layout
            bottom_margin = height // 8 if hasattr(params, 'horizontal') and params.horizontal else margin
            if y_line_position + font_size > height - bottom_margin:
                break
            
            # Render each word in the line
            x_word_position = x_position
            for word_index, word in enumerate(line):
                # Use enhanced text layout if enabled
                if hasattr(params, 'enhanced_layout') and params.enhanced_layout:
                    # Get character positions with enhanced layout
                    char_positions = _enhance_text_layout(
                        draw, word, font, x_word_position, y_line_position, 
                        line_index, width, font_size, params
                    )
                    
                    # Render each character individually
                    for i, char in enumerate(word):
                        px, py = char_positions[i]
                        
                        # Add color variation for natural ink appearance
                        color_variation = int(random.randint(-3, 3) * params.ink_color)
                        char_color = (
                            np.clip(ink_color[0] + color_variation, 0, 255),
                            np.clip(ink_color[1] + color_variation, 0, 255),
                            np.clip(ink_color[2] + color_variation, 0, 255)
                        )
                        
                        draw.text((px, py), char, fill=char_color, font=font)
                    
                    # Calculate where the next word should start
                    word_width = draw.textlength(word, font=font)
                    x_word_position += word_width + space_width
                else:
                    # Original word rendering logic
                    word_x_offset = int(random.uniform(-1.5, 1.5) * params.word_position)
                    word_y_offset = int(random.uniform(-1, 1) * params.word_position)
                    
                    color_variation = int(random.randint(-3, 3) * params.ink_color)
                    word_color = (
                        np.clip(ink_color[0] + color_variation, 0, 255),
                        np.clip(ink_color[1] + color_variation, 0, 255),
                        np.clip(ink_color[2] + color_variation, 0, 255)
                    )
                    
                    word_width = draw.textlength(word, font=font)
                    word_height = font_size * 1.2
                    
                    if params.word_angle > 0:
                        # Apply rotation to individual word
                        word_angle = random.uniform(-2, 2) * params.word_angle
                        
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
            
            # Move to next line with reduced randomness for more consistent spacing
            line_spacing_factor = 1.0 + (random.uniform(-0.05, 0.05) * params.line_spacing)
            y_position += int(font_size * 1.2 * line_spacing_factor)
        
        img.save(output_path)
        print(f"Saved rendered Sanskrit to {output_path}")
        return img
        
    except Exception as e:
        print(f"Error rendering text with font {font_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

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
                                            random.uniform(-params.rotation_max, params.rotation_max))))
    transforms.append(("brightness", lambda img: adjust_brightness(img, 
                                            random.uniform(1.0-params.brightness_var, 1.0+params.brightness_var))))
    transforms.append(("contrast", lambda img: adjust_contrast(img, 
                                            random.uniform(1.0-params.contrast_var, 1.0+params.contrast_var))))
    transforms.append(("noise", lambda img: add_noise(img, 
                                            random.uniform(params.noise_min, params.noise_max))))
    transforms.append(("blur", lambda img: blur_image(img, 
                                            random.uniform(params.blur_min, params.blur_max))))
    
    if params.all_transforms:
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

def _generate_sanskrit_samples(text, font_path, output_dir, params):
    if not os.path.exists(font_path):
        print(f"Error: Font not found at {font_path}")
        return
    
    styles = ["lined_paper", "old_paper", "birch", "parchment"]
    
    ink_colors = {
        "lined_paper": (60, 30, 10),
        "old_paper": (20, 20, 20),
        "birch": (50, 20, 10),
        "parchment": (10, 10, 10)
    }
    
    width, height = params.width, params.height
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly sample styles for the total number of base images
    sampled_styles = random.choices(styles, k=params.base_images)
    style_counts = {style: sampled_styles.count(style) for style in styles}
    print(f"Randomly selected styles: {style_counts}")
    
    base_images = []
    image_counter = 0
    
    # Generate randomly sampled base images
    for style, count in style_counts.items():
        for i in range(count):
            image_counter += 1
            output_path = os.path.join(output_dir, f"sanskrit_{style}_{i+1}.png")
            
            # Randomly select a font size between 12 and 18
            font_size = random.randint(12, 18)
            print(f"Using font size {font_size} for {style}_{i+1}")
            
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
                
                if params.apply_transforms:
                    base_filename = f"sanskrit_{style}_{i+1}"
                    _apply_postprocessing(img, output_dir, base_filename, params)

def main():
    sanskrit_text = """ज्ञानं परमं ध्येयम्। ज्ञानात् सत्यं प्रकाशते। सत्येन मुक्तिः प्राप्यते। मुक्तिः परमं सुखम्। तस्मात् ज्ञानं समभ्यसेत्। विद्या ददाति विनयम्। विनयात् याति पात्रताम्। पात्रत्वात् धनमाप्नोति। धनात् धर्मं ततः सुखम्॥"""
    
    parser = argparse.ArgumentParser(description='Generate Sanskrit text samples with word-level and image augmentations')
    
    # Basic options
    basic = parser.add_argument_group('Basic Options')
    basic.add_argument('--output-dir', type=str, default='data/synthetic/images',
                      help='Output directory for generated images')
    basic.add_argument('--width', type=int, default=400,
                      help='Width of output images')
    basic.add_argument('--height', type=int, default=320, 
                      help='Height of output images')
    basic.add_argument('--base-images', type=int, default=5,
                      help='Total number of base images to generate (randomly sampled styles)')
    
    # Font options
    font = parser.add_argument_group('Font Options')
    font.add_argument('--font-dir', type=str, default=r'C:\Users\Rahul Badhan\Documents\GitHub\sanskrit-ocr\datagen\fonts',
                    help='Directory containing font files')
    font.add_argument('--font', type=str, default=r'C:\Users\Rahul Badhan\Documents\GitHub\sanskrit-ocr\datagen\fonts\Sharad76-Regular.otf',
                    help='Font filename within the font directory')
    
    # Layout options (new group)
    layout = parser.add_argument_group('Layout Options', 
                                     'Controls for manuscript layout')
    layout.add_argument('--horizontal', action='store_true', default=False,
                       help='Use horizontal manuscript layout (default: False)')
    layout.add_argument('--vertical-lines', action='store_true', default=True,
                       help='Add vertical ruling lines for horizontal layout (default: True)')
    layout.add_argument('--enhanced-layout', action='store_true', default=False,
                       help='Use enhanced text layout with natural character spacing (default: False)')
    
    # Generation-level augmentations (background + word)
    gen = parser.add_argument_group('Generation-Level Augmentations', 
                                  'Controls for background and word-level effects')
    # Background options
    gen.add_argument('--noise', type=float, default=0.7,
                   help='Background noise intensity (0.0-1.0)')
    gen.add_argument('--aging', type=float, default=0.6,
                   help='Edge aging effect (0.0-1.0)')
    gen.add_argument('--texture', type=float, default=0.7,
                   help='Texture variation (0.0-1.0)')
    gen.add_argument('--stains', type=float, default=0.6,
                   help='Number of stains (0.0-1.0)')
    gen.add_argument('--stain-intensity', type=float, default=0.5,
                   help='Intensity of stain effects (0.0-1.0)')
    
    # Enhanced degradation options (new)
    gen.add_argument('--insect-damage', type=float, default=0.0,
                   help='Intensity of insect damage (0.0-1.0)')
    gen.add_argument('--water-damage', type=float, default=0.0,
                   help='Intensity of water damage (0.0-1.0)')
    gen.add_argument('--edge-deterioration', type=float, default=0.0,
                   help='Edge deterioration intensity (0.0-1.0)')
    
    # Word-level options
    gen.add_argument('--word-position', type=float, default=0.6,
                   help='Random word position variation (0.0-1.0)')
    gen.add_argument('--ink-color', type=float, default=0.5,
                   help='Ink color variation (0.0-1.0)')
    gen.add_argument('--line-spacing', type=float, default=0.4,
                   help='Random line spacing (0.0-1.0)')
    gen.add_argument('--baseline', type=float, default=0.3,
                   help='Baseline wobble effect (0.0-1.0)')
    gen.add_argument('--word-angle', type=float, default=0.0,
                   help='Random word angle (0.0-1.0)')
    
    # Post-processing options (image transforms)
    post = parser.add_argument_group('Post-Processing Augmentations', 
                                   'Controls for image transformations after rendering')
    post.add_argument('--no-transforms', dest='apply_transforms', action='store_false',
                    help='Disable post-processing transforms')
    post.add_argument('--all-transforms', action='store_true',
                    help='Apply all transforms instead of random subset')
    post.add_argument('--rotation-max', type=float, default=5.0,
                    help='Maximum rotation angle in degrees')
    post.add_argument('--brightness-var', type=float, default=0.2,
                    help='Brightness variation factor (0.0-1.0)')
    post.add_argument('--contrast-var', type=float, default=0.2,
                    help='Contrast variation factor (0.0-1.0)')
    post.add_argument('--noise-min', type=float, default=0.01,
                    help='Minimum noise intensity for transforms')
    post.add_argument('--noise-max', type=float, default=0.05,
                    help='Maximum noise intensity for transforms')
    post.add_argument('--blur-min', type=float, default=0.5,
                    help='Minimum blur radius')
    post.add_argument('--blur-max', type=float, default=1.0,
                    help='Maximum blur radius')
    
    parser.set_defaults(apply_transforms=True, all_transforms=False)
    params = parser.parse_args()
    
    font_path = os.path.join(params.font_dir, params.font)
    
    # Print layout information if horizontal is enabled
    if params.horizontal:
        print("Using horizontal manuscript layout")
        if params.vertical_lines:
            print("Adding vertical ruling lines")
    
    _generate_sanskrit_samples(
        text=sanskrit_text,
        font_path=font_path,
        output_dir=params.output_dir,
        params=params
    )

if __name__ == "__main__":
    main()