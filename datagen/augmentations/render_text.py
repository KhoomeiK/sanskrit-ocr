import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import os
import sys
import random
import math
import argparse

AUGMENTATION_COEFFS = {
    # Background texture effects (0.0 = none, 1.0 = full effect)
    "noise_intensity": 0.7,      
    "edge_aging": 0.6,           
    "stain_intensity": 0.5,      
    "stain_count": 0.6,          
    "texture_variation": 0.7,    
    
    # Text rendering effects
    "word_position_random": 0.6, 
    "ink_color_variation": 0.5,  
    "line_spacing_random": 0.4,  
    "baseline_wobble": 0.3,      
    "word_angle_random": 0.0,    
}

# Sanskrit text to render
sanskrit_text = """ज्ञानं परमं ध्येयम्। ज्ञानात् सत्यं प्रकाशते।
सत्येन मुक्तिः प्राप्यते। मुक्तिः परमं सुखम्।
तस्मात् ज्ञानं समभ्यसेत्। विद्या ददाति विनयम्।
विनयात् याति पात्रताम्। पात्रत्वात् धनमाप्नोति।
धनात् धर्मं ततः सुखम्॥"""

def _create_background(width, height, style, coeffs=None):
    """Create a realistic manuscript background"""
    # Use provided coefficients or defaults
    coeffs = coeffs or AUGMENTATION_COEFFS
    
    if style == "palm_leaf":
        background = np.ones((height, width, 3), dtype=np.uint8) * [210, 180, 140]
        
        line_spacing = random.randint(15, 25)
        for y in range(0, height, line_spacing):
            line_width = random.randint(1, 2)
            darkness = random.randint(6, 20) * coeffs["texture_variation"]
            background[y:y+line_width, :, :] = np.clip(background[y:y+line_width, :, :] - darkness, 0, 255)
            
        noise = np.random.randint(0, int(15 * coeffs["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        stain_count = int(random.randint(2, 4) * coeffs["stain_count"])
        for _ in range(stain_count):
            x = random.randint(0, width-100)
            y = random.randint(0, height-100)
            size = random.randint(20, 60)
            darkness = random.randint(8, 25) * coeffs["stain_intensity"]
            shape = np.ones((size, size, 3), dtype=np.uint8) * darkness
            # Gaussian-like pattern
            for i in range(size):
                for j in range(size):
                    dist = ((i - size/2)**2 + (j - size/2)**2) / (size/4)**2
                    if dist < 1:
                        alpha = (1 - dist) * random.uniform(0.4, 0.8) * coeffs["stain_intensity"]
                        if y+i < height and x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] - shape[i, j, :] * alpha, 0, 255
                            )
    
    elif style == "old_paper":
        # Create aged paper texture (yellowish)
        background = np.ones((height, width, 3), dtype=np.uint8) * [236, 222, 181]
        
        # Add noise for paper texture
        noise = np.random.randint(0, int(12 * coeffs["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        # Add some yellowing around edges
        edge_width = width // 10
        for i in range(edge_width):
            factor = (edge_width - i) / edge_width * 15 * coeffs["edge_aging"]
            # Top edge
            background[i, :, 2] = np.clip(background[i, :, 2] - factor, 0, 255)
            # Bottom edge
            background[height-i-1, :, 2] = np.clip(background[height-i-1, :, 2] - factor, 0, 255)
            # Left edge
            background[:, i, 2] = np.clip(background[:, i, 2] - factor, 0, 255)
            # Right edge
            background[:, width-i-1, 2] = np.clip(background[:, width-i-1, 2] - factor, 0, 255)
    
    elif style == "birch_bark":
        # Create birch bark texture
        background = np.ones((height, width, 3), dtype=np.uint8) * [235, 225, 215]
        
        noise = np.random.randint(0, int(10 * coeffs["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        # Add some subtle texture variations
        variation_count = int(150 * coeffs["texture_variation"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(10, 25)
            variation = random.randint(-6, 6) * coeffs["texture_variation"]
            
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
    
    else:  # "parchment"
        # Create parchment texture
        background = np.ones((height, width, 3), dtype=np.uint8) * [230, 215, 185]
        
        # Add texture variations
        variation_count = int(400 * coeffs["texture_variation"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 12)
            variation = random.randint(-7, 7) * coeffs["texture_variation"]
            
            # Some circular patches
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
        
        # Add noise
        noise = np.random.randint(0, int(8 * coeffs["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(background)

def render_sanskrit(text, font_path, output_path, 
                   width=500, height=400, 
                   font_size=30, 
                   background_style="parchment",
                   ink_color=(0, 0, 0),
                   coeffs=None):
    """Render Sanskrit text with the given font and background style - word by word processing"""
    
    # Use provided coefficients or defaults
    coeffs = coeffs or AUGMENTATION_COEFFS
    
    # Create background
    img = _create_background(width, height, background_style, coeffs)
    draw = ImageDraw.Draw(img)
    
    try:
        # Load font
        font = ImageFont.truetype(font_path, font_size)
        
        # Draw text line by line
        lines = text.strip().split('\n')
        y_position = 30  # Starting position
        
        for line in lines:
            # Add slight randomness for handwritten feel to baseline
            baseline_offset = random.randint(-2, 2) * coeffs["baseline_wobble"]
            y_line_position = y_position + baseline_offset
            
            # Process the line word by word
            words = line.split()
            x_position = (width - draw.textlength(line, font=font)) // 2
            
            for word in words:
                # For each word, apply slight position variations
                word_x_offset = int(random.uniform(-1.5, 1.5) * coeffs["word_position_random"])
                word_y_offset = int(random.uniform(-1, 1) * coeffs["word_position_random"])
                
                # For handwritten look, vary darkness slightly
                color_variation = int(random.randint(-3, 3) * coeffs["ink_color_variation"])
                word_color = (
                    np.clip(ink_color[0] + color_variation, 0, 255),
                    np.clip(ink_color[1] + color_variation, 0, 255),
                    np.clip(ink_color[2] + color_variation, 0, 255)
                )
                
                # Get the word dimensions for proper spacing
                if hasattr(draw, 'textlength'):
                    word_width = draw.textlength(word, font=font)
                    # For height, we use an approximation
                    word_height = font_size * 1.2
                else:
                    word_width, word_height = font.getsize(word)
                
                # Calculate rotation if needed
                if coeffs["word_angle_random"] > 0:
                    word_angle = random.uniform(-2, 2) * coeffs["word_angle_random"]
                    
                    # Calculate the size needed for the rotated word
                    # The diagonal of the word's bounding box is the maximum possible width/height after rotation
                    diagonal = math.sqrt(word_width**2 + word_height**2)
                    # Add some padding to ensure the entire rotated word fits
                    padding = int(diagonal * 0.5)
                    
                    # Create a transparent image with enough space for the rotated word
                    temp_width = int(diagonal + 2 * padding)
                    temp_height = int(diagonal + 2 * padding)
                    txt_img = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
                    txt_d = ImageDraw.Draw(txt_img)
                    
                    # Draw word on transparent image, centered
                    center_x = temp_width // 2 - word_width // 2
                    center_y = temp_height // 2 - word_height // 2
                    txt_d.text((center_x, center_y), word, font=font, fill=word_color + (255,))
                    
                    # Rotate the word around its center
                    rotated = txt_img.rotate(word_angle, resample=Image.BICUBIC, expand=0, center=(temp_width//2, temp_height//2))
                    
                    # Calculate paste position to center the rotated word on the line
                    paste_x = int(x_position + word_x_offset - padding)
                    paste_y = int(y_line_position + word_y_offset - padding)
                    
                    # Paste the rotated word onto the main image
                    img.paste(rotated, (paste_x, paste_y), rotated)
                else:
                    # Draw word directly if no rotation
                    draw.text(
                        (x_position + word_x_offset, y_line_position + word_y_offset), 
                        word, 
                        fill=word_color, 
                        font=font
                    )
                
                # Move the position for the next word (add a space)
                x_position += word_width + draw.textlength(" ", font=font)
            
            # Add slight randomness to line spacing
            line_spacing_factor = 1.0 + (random.uniform(-0.1, 0.1) * coeffs["line_spacing_random"])
            y_position += int(font_size * 1.2 * line_spacing_factor)
        
        # Save the image
        img.save(output_path)
        print(f"Saved rendered Sanskrit to {output_path}")
        return img
        
    except Exception as e:
        print(f"Error rendering text with font {font_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def _apply_image_transforms(original_image, output_dir, base_filename, apply_all=False):
    """Apply various transformations to an image and save them to the output directory."""
    all_images = [original_image]
    transforms = []
    
    width, height = original_image.size
    
    # Define transforms
    def rotate_image(img, angle):
        bg_color = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)
        return rotated
    
    transforms.append(("rotate", lambda img: rotate_image(img, random.uniform(-5, 5))))
    
    def adjust_brightness(img, factor):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    transforms.append(("brightness", lambda img: adjust_brightness(img, random.uniform(0.8, 1.2))))
    
    def adjust_contrast(img, factor):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    transforms.append(("contrast", lambda img: adjust_contrast(img, random.uniform(0.8, 1.2))))
    
    def add_noise(img, intensity=0.05):
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    transforms.append(("noise", lambda img: add_noise(img, random.uniform(0.01, 0.05))))
    
    def blur_image(img, radius):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    transforms.append(("blur", lambda img: blur_image(img, random.uniform(0.5, 1.0))))
    
    if apply_all:
        selected_transforms = transforms
    else:
        n_transforms = random.randint(1, 3)
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

def _generate_sanskrit_samples(text=None, font_path=None, coeffs=None, show_plot=False, apply_transforms=True):
    """Generate samples with only the Sharad font in a compact grid"""
    
    text = text or sanskrit_text
    
    default_font_path = os.path.join("datagen/fonts", "Sharad76-Regular.otf")
    font_path = font_path or default_font_path
    
    if not os.path.exists(font_path):
        print(f"Error: Font not found at {font_path}")
        return
    
    styles = ["palm_leaf", "old_paper", "birch_bark", "parchment"]
    
    ink_colors = {
        "palm_leaf": (60, 30, 10),    # Dark brown
        "old_paper": (20, 20, 20),    # Nearly black
        "birch_bark": (50, 20, 10),   # Reddish brown
        "parchment": (10, 10, 10)     # Black
    }
    
    font_sizes = {
        "palm_leaf": 14,
        "old_paper": 16,
        "birch_bark": 12,
        "parchment": 18
    }
    
    rendered_images = []
    all_generated_images = []
    
    width, height = 400, 320
    
    output_dir = os.path.join("data", "synthetic", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    transforms_dir = os.path.join(output_dir, "transforms")
    if apply_transforms:
        os.makedirs(transforms_dir, exist_ok=True)
    
    for style in styles:
        output_path = os.path.join(output_dir, f"sanskrit_{style}.png")
        
        img = render_sanskrit(
            text=text,
            font_path=font_path,
            output_path=output_path,
            width=width,
            height=height,
            font_size=font_sizes[style],
            background_style=style,
            ink_color=ink_colors[style],
            coeffs=coeffs
        )
        
        if img:
            rendered_images.append(img)
            all_generated_images.append(img)
            
            if apply_transforms:
                base_filename = f"sanskrit_{style}"
                transformed_images = _apply_image_transforms(img, transforms_dir, base_filename)
                all_generated_images.extend(transformed_images[1:])  # Skip the original
    
    if rendered_images:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        plt.subplots_adjust(wspace=0, hspace=0)
        
        axes = axes.flatten()
        
        for i, img in enumerate(rendered_images):
            axes[i].imshow(np.array(img))
            axes[i].axis('off')
        
        plt.tight_layout(pad=0)
        grid_output_path = os.path.join(output_dir, "sanskrit_all_samples.png")
        plt.savefig(grid_output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"All samples saved to {grid_output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        if apply_transforms and len(all_generated_images) > 4:
            n_images = len(all_generated_images)
            grid_cols = min(5, n_images)
            grid_rows = (n_images + grid_cols - 1) // grid_cols
            
            fig2, axes2 = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3, grid_rows*3))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            
            if grid_rows > 1 or grid_cols > 1:
                axes2 = axes2.flatten()
            
            for i, img in enumerate(all_generated_images):
                if grid_rows == 1 and grid_cols == 1:
                    ax = axes2
                else:
                    ax = axes2[i]
                ax.imshow(np.array(img))
                ax.axis('off')
                
            if grid_rows > 1 or grid_cols > 1:
                for j in range(n_images, grid_rows*grid_cols):
                    axes2[j].axis('off')
                    axes2[j].set_visible(False)
            
            all_transforms_path = os.path.join(output_dir, "sanskrit_all_transforms.png")
            plt.savefig(all_transforms_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            print(f"All transforms grid saved to {all_transforms_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig2)
    else:
        print("Failed to generate any samples.")

def main(text=None, font_path=None, coeffs=None, show_plot=False, apply_transforms=True):
    """Main function for generating Sanskrit text samples"""
    _generate_sanskrit_samples(text=text, font_path=font_path, coeffs=coeffs, show_plot=show_plot, apply_transforms=apply_transforms)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Sanskrit text samples')
    parser.add_argument('--show', action='store_true', help='Show the plot window')
    parser.add_argument('--no-transforms', dest='transforms', action='store_false', help='Disable image transforms')
    parser.set_defaults(transforms=True)
    args = parser.parse_args()
    
    main(show_plot=args.show, apply_transforms=args.transforms)