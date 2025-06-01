import os
import sys
import random
import importlib.util
from pathlib import Path
from tqdm import tqdm
from typing import Iterator, List, Tuple
from PIL import Image
from augmentations.effects import apply_effects


class RenderingManager:
    def __init__(self):
        self.renderers = {}
        self._load_renderers()
    
    def _load_renderers(self):
        """Dynamically load all render_*.py files from the rendering folder."""
        rendering_dir = Path(__file__).parent / "rendering"
        
        # Find all render_*.py files
        for render_file in rendering_dir.glob("render_*.py"):
            module_name = render_file.stem
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, render_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Check if module has a render function
            if hasattr(module, 'render'):
                self.renderers[module_name] = module.render
                print(f"Loaded renderer: {module_name}")
            else:
                print(f"Warning: {module_name} does not have a render function")
    
    def get_random_renderer(self):
        """Get a random render function from available renderers."""
        if not self.renderers:
            raise RuntimeError("No renderers available")
        return random.choice(list(self.renderers.values()))
    
    def render_random(self, text: str, use_max: bool = False) -> Tuple[Image.Image, str]:
        """Render text using a randomly selected renderer."""
        renderer = self.get_random_renderer()
        img, caption = renderer(text, use_max=use_max)
        
        # Apply effects
        img = apply_effects(img)
        
        return img, caption


def chunk_text(lines_iter: Iterator[str], min_chars: int = 1050, max_chars: int = 1350) -> Iterator[str]:
    """
    Chunk lines from iterator into strings of specified character length.
    
    Args:
        lines_iter: Iterator yielding lines of text
        min_chars: Minimum characters per chunk
        max_chars: Maximum characters per chunk
    
    Yields:
        Chunked text strings
    """
    current_chunk = []
    current_length = 0
    target_length = random.randint(min_chars, max_chars)
    
    for line in lines_iter:
        line = line.strip()
        if not line:
            continue
            
        line_length = len(line)
        
        # If adding this line would exceed our target, yield current chunk
        if current_length + line_length > target_length and current_chunk:
            yield "\n\n".join(current_chunk)
            current_chunk = []
            current_length = 0
            target_length = random.randint(min_chars, max_chars)
        
        current_chunk.append(line)
        current_length += line_length + 2  # +2 for the \n\n separator
    
    # Don't forget the last chunk
    if current_chunk:
        yield "\n\n".join(current_chunk)


def generate_dataset(data: Iterator[str], num_samples: int = None, use_max: bool = False, images_per_sample: int = 1) -> List[Tuple[Image.Image, str]]:
    """
    Generate dataset of images and captions from text data.
    
    Args:
        data: Iterator yielding lines of text
        num_samples: Maximum number of samples to generate (None for all)
        use_max: Whether to use maximum sizing parameters for rendering
        images_per_sample: Number of images to generate per text chunk
    
    Returns:
        List of (PIL.Image, caption_text) tuples
    """
    manager = RenderingManager()
    results = []
    
    chunks = chunk_text(data)
    
    for i, chunk in enumerate(tqdm(chunks, desc="Chunks")):
        if num_samples and i >= num_samples:
            break
        
        for j in tqdm(range(images_per_sample), desc="Samples", leave=False):
            try:
                img, caption = manager.render_random(chunk, use_max=use_max)
                results.append((img, caption))
            except Exception as e:
                print(f"Error rendering sample {i}-{j}: {e}")
                continue
    
    return results

# Convenience function for backwards compatibility
def render_random(text: str, use_max: bool = False) -> Tuple[Image.Image, str]:
    """Render text using a randomly selected renderer."""
    manager = RenderingManager()
    img, caption = manager.render_random(text, use_max)
    return img, caption