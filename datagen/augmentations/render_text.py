import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import sys
import random
import math

AUGMENTATION_COEFFS = {
    # Background texture effects (0.0 = none, 1.0 = full effect)
    "noise_intensity": 0.7,      # Controls random noise in backgrounds
    "edge_aging": 0.6,           # Controls edge yellowing/darkening
    "stain_intensity": 0.5,      # Controls darkness and size of stains/patches
    "stain_count": 0.6,          # Controls number of stains
    "texture_variation": 0.7,    # Controls amount of texture variation
    
    # Text rendering effects
    "word_position_random": 0.6, # Controls random position offsets of words
    "ink_color_variation": 0.5,  # Controls variation in ink color
    "line_spacing_random": 0.4,  # Controls randomness in line spacing
    "baseline_wobble": 0.3,      # Controls variation in baseline position
    "word_angle_random": 0.0,    # Doesn't work as intended currently
}

# Sanskrit text to render
sanskrit_text = """ज्ञानं परमं ध्येयम्। ज्ञानात् सत्यं प्रकाशते।
सत्येन मुक्तिः प्राप्यते। मुक्तिः परमं सुखम्।
तस्मात् ज्ञानं समभ्यसेत्। विद्या ददाति विनयम्।
विनयात् याति पात्रताम्। पात्रत्वात् धनमाप्नोति।
धनात् धर्मं ततः सुखम्॥"""

def create_background(width, height, style):
    """Create a realistic manuscript background"""
    if style == "palm_leaf":
        background = np.ones((height, width, 3), dtype=np.uint8) * [210, 180, 140]
        
        line_spacing = random.randint(15, 25)
        for y in range(0, height, line_spacing):
            line_width = random.randint(1, 2)
            darkness = random.randint(6, 20) * AUGMENTATION_COEFFS["texture_variation"]
            background[y:y+line_width, :, :] = np.clip(background[y:y+line_width, :, :] - darkness, 0, 255)
            
        noise = np.random.randint(0, int(15 * AUGMENTATION_COEFFS["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        stain_count = int(random.randint(2, 4) * AUGMENTATION_COEFFS["stain_count"])
        for _ in range(stain_count):
            x = random.randint(0, width-100)
            y = random.randint(0, height-100)
            size = random.randint(20, 60)
            darkness = random.randint(8, 25) * AUGMENTATION_COEFFS["stain_intensity"]
            shape = np.ones((size, size, 3), dtype=np.uint8) * darkness
            # Gaussian-like pattern
            for i in range(size):
                for j in range(size):
                    dist = ((i - size/2)**2 + (j - size/2)**2) / (size/4)**2
                    if dist < 1:
                        alpha = (1 - dist) * random.uniform(0.4, 0.8) * AUGMENTATION_COEFFS["stain_intensity"]
                        if y+i < height and x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] - shape[i, j, :] * alpha, 0, 255
                            )
    
    elif style == "old_paper":
        # Create aged paper texture (yellowish)
        background = np.ones((height, width, 3), dtype=np.uint8) * [236, 222, 181]
        
        # Add noise for paper texture
        noise = np.random.randint(0, int(12 * AUGMENTATION_COEFFS["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        # Add some yellowing around edges
        edge_width = width // 10
        for i in range(edge_width):
            factor = (edge_width - i) / edge_width * 15 * AUGMENTATION_COEFFS["edge_aging"]
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
        
        noise = np.random.randint(0, int(10 * AUGMENTATION_COEFFS["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
        
        # Add some subtle texture variations
        variation_count = int(150 * AUGMENTATION_COEFFS["texture_variation"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(10, 25)
            variation = random.randint(-6, 6) * AUGMENTATION_COEFFS["texture_variation"]
            
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
        variation_count = int(400 * AUGMENTATION_COEFFS["texture_variation"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 12)
            variation = random.randint(-7, 7) * AUGMENTATION_COEFFS["texture_variation"]
            
            # Some circular patches
            for i in range(-size, size):
                for j in range(-size, size):
                    if i*i + j*j <= size*size:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            background[y+i, x+j, :] = np.clip(
                                background[y+i, x+j, :] + variation, 0, 255
                            )
        
        # Add noise
        noise = np.random.randint(0, int(8 * AUGMENTATION_COEFFS["noise_intensity"]), 
                                (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(background)

def render_sanskrit(text, font_path, output_path, 
                   width=500, height=400, 
                   font_size=30, 
                   background_style="parchment",
                   ink_color=(0, 0, 0)):
    """Render Sanskrit text with the given font and background style - word by word processing"""
    
    # Create background
    img = create_background(width, height, background_style)
    draw = ImageDraw.Draw(img)
    
    try:
        # Load font
        font = ImageFont.truetype(font_path, font_size)
        
        # Draw text line by line
        lines = text.strip().split('\n')
        y_position = 30  # Starting position
        
        for line in lines:
            # Add slight randomness for handwritten feel to baseline
            baseline_offset = random.randint(-2, 2) * AUGMENTATION_COEFFS["baseline_wobble"]
            y_line_position = y_position + baseline_offset
            
            # Process the line word by word
            words = line.split()
            x_position = (width - draw.textlength(line, font=font)) // 2
            
            for word in words:
                # For each word, apply slight position variations
                word_x_offset = int(random.uniform(-1.5, 1.5) * AUGMENTATION_COEFFS["word_position_random"])
                word_y_offset = int(random.uniform(-1, 1) * AUGMENTATION_COEFFS["word_position_random"])
                
                # For handwritten look, vary darkness slightly
                color_variation = int(random.randint(-3, 3) * AUGMENTATION_COEFFS["ink_color_variation"])
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
                if AUGMENTATION_COEFFS["word_angle_random"] > 0:
                    word_angle = random.uniform(-2, 2) * AUGMENTATION_COEFFS["word_angle_random"]
                    
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
            line_spacing_factor = 1.0 + (random.uniform(-0.1, 0.1) * AUGMENTATION_COEFFS["line_spacing_random"])
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

def generate_sanskrit_samples():
    """Generate samples with only the Sharad font in a compact grid"""
    
    # Font path
    sharad_font_path = os.path.join("fonts", "Sharad76-Regular.otf")
    
    # Check if local Sharad font exists
    if not os.path.exists(sharad_font_path):
        print(f"Error: Sharad76-Regular.otf not found in the fonts directory")
        return
    
    # Background styles
    styles = ["palm_leaf", "old_paper", "birch_bark", "parchment"]
    
    # Ink colors for different materials
    ink_colors = {
        "palm_leaf": (60, 30, 10),    # Dark brown
        "old_paper": (20, 20, 20),    # Nearly black
        "birch_bark": (50, 20, 10),   # Reddish brown
        "parchment": (10, 10, 10)     # Black
    }
    
    # Font sizes for different materials
    font_sizes = {
        "palm_leaf": 14,
        "old_paper": 16,
        "birch_bark": 12,
        "parchment": 18
    }
    
    # Generate samples
    rendered_images = []
    
    # Sample size - smaller for more compact grid
    width, height = 400, 320
    
    for style in styles:
        output_path = f"sanskrit_{style}.png"
        
        img = render_sanskrit(
            text=sanskrit_text,
            font_path=sharad_font_path,
            output_path=output_path,
            width=width,
            height=height,
            font_size=font_sizes[style],
            background_style=style,
            ink_color=ink_colors[style]
        )
        
        if img:
            rendered_images.append(img)
    
    # Combine all images into a compact grid
    if rendered_images:
        # Create figure with no padding/spacing
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        plt.subplots_adjust(wspace=0, hspace=0)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Add each image to the grid without titles
        for i, img in enumerate(rendered_images):
            axes[i].imshow(np.array(img))
            axes[i].axis('off')  # Turn off axes
        
        # Remove padding around the entire figure
        plt.tight_layout(pad=0)
        plt.savefig("sanskrit_all_samples.png", bbox_inches='tight', pad_inches=0, dpi=150)
        print("All samples saved to sanskrit_all_samples.png")
        plt.show()
    else:
        print("Failed to generate any samples.")

if __name__ == "__main__":
    generate_sanskrit_samples()