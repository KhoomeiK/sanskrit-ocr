import random
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image

from render_text import generate_sanskrit_samples
from render_book_page import generate_book_pages
from render_parchment_leaf import generate_parchment_leaves

# Default parameters
DEFAULT_PARAMS = {
    'output_dir': 'output_random',
    'renderer_weights': [0.33, 0.33, 0.34],  # Equal weights for each renderer
    'renderers': [
        ('text', generate_sanskrit_samples),
        ('book', generate_book_pages),
        ('parchment', generate_parchment_leaves)
    ]
}

def generate_random_samples(
    text: str,
    output_dir: Optional[str] = None,
    params: Optional[dict] = None,
    num_samples: int = 1
) -> Union[List[Image.Image], None]:
    """
    Generate random samples using any of the three renderers.
    
    Args:
        text (str): The Sanskrit text to render
        output_dir (str, optional): Directory to save images. If None, images are returned in memory
        params (dict, optional): Parameters to override defaults
        num_samples (int): Number of samples to generate
        
    Returns:
        If output_dir is None: list of PIL.Image objects
        If output_dir is provided: None (images are saved to disk)
    """
    # Use default params if none provided, updating with any provided values
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        params = {**DEFAULT_PARAMS, **params}
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    all_images = []
    
    for i in range(num_samples):
        # Randomly select a renderer
        renderer_name, renderer_func = random.choices(
            params['renderers'],
            weights=params['renderer_weights'],
            k=1
        )[0]
        
        # Create subdirectory for this renderer if output_dir is provided
        if output_dir:
            renderer_dir = output_dir / f"{i:03d}_{renderer_name}"
            renderer_dir.mkdir(exist_ok=True)
        else:
            renderer_dir = None
        
        # Generate images using selected renderer
        images = renderer_func(text, output_dir=renderer_dir)
        
        if not output_dir:
            all_images.extend(images)
    
    return all_images if not output_dir else None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random Sanskrit text samples')
    parser.add_argument('--num', type=int, default=10,
                      help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PARAMS['output_dir'],
                      help='Output directory for generated images')
    args = parser.parse_args()
    
    # Read sample text
    text = Path("sample_sa.txt").read_text("utf-8").strip()
    
    # Generate random samples
    generate_random_samples(
        text=text,
        output_dir=args.output_dir,
        num_samples=args.num
    ) 