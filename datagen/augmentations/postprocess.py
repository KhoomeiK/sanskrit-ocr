import random
from PIL import Image


def rotate(img: Image.Image, angle: float = None) -> Image.Image:
    """Rotate image by a random angle."""
    if angle is None:
        angle = random.uniform(-3, 3)
    return img.rotate(angle)


EFFECTS = {
    "rotate": (
        rotate, 
        1.0,
        lambda: {"angle": random.uniform(-3, 3)}
    ),
}


def build_effect_chain():
    """Build a chain of effects based on probabilities."""
    chain = []
    for name, (func, prob, param_gen) in EFFECTS.items():
        if random.random() < prob:
            params = param_gen() if param_gen else {}
            chain.append((func, params))
    return chain


def apply_effects(img: Image.Image) -> Image.Image:
    """Apply a random chain of effects to an image."""
    chain = build_effect_chain()
    
    for func, params in chain:
        img = func(img, **params)
    
    return img
