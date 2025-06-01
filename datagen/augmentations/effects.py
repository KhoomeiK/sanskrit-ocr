import random
import numpy as np
from PIL import Image
import cv2
from typing import Callable, Dict, List, Tuple
from augraphy import (
    BadPhotoCopy,
    LightingGradient,
    PageBorder,
    Folding,
    ShadowCast,
    VoronoiTessellation,
    ReflectedLight
)

NO_EFFECT = 0.1

def rotate(img: Image.Image, angle: float = None) -> Image.Image:
    if angle is None:
        angle = random.uniform(-3, 2)
    return img.rotate(angle)


def bad_photocopy(img: Image.Image) -> Image.Image:
    aug = BadPhotoCopy()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def lighting_gradient(img: Image.Image) -> Image.Image:
    aug = LightingGradient()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def page_border(img: Image.Image) -> Image.Image:
    aug = PageBorder()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def folding(img: Image.Image) -> Image.Image:
    aug = Folding()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def shadow_cast(img: Image.Image) -> Image.Image:
    aug = ShadowCast()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def voronoi_tessellation(img: Image.Image) -> Image.Image:
    aug = VoronoiTessellation()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def reflected_light(img: Image.Image) -> Image.Image:
    aug = ReflectedLight()
    img_np = np.array(img)
    result = aug(image=img_np)
    return Image.fromarray(result)


def paper_stains_and_damage(img: Image.Image) -> Image.Image:
    from augraphy.utilities.texturegenerator import TextureGenerator

    img_np = np.array(img)
    ysize, xsize = img_np.shape[:2]

    texture_generator = TextureGenerator()
    stain_type = random.choice([
        "fine_stains",
        "severe_stains",
        "light_stains"
    ])
    stains = texture_generator(
        texture_type=stain_type,
        texture_width=xsize,
        texture_height=ysize,
        quilt_texture=0,
    )

    edge_type = random.choice(["curvy_edge", "broken_edge"])
    edge_damage = texture_generator(
        texture_type=edge_type,
        texture_width=xsize,
        texture_height=ysize,
        quilt_texture=0,
    )

    if len(img_np.shape) == 3:
        if len(stains.shape) == 2:
            stains = cv2.cvtColor(stains, cv2.COLOR_GRAY2BGR)
        if len(edge_damage.shape) == 2:
            edge_damage = cv2.cvtColor(edge_damage, cv2.COLOR_GRAY2BGR)

    img_float = img_np.astype(float)
    stains_float = stains.astype(float) / 255.0
    stained = img_float * stains_float

    if edge_type == "curvy_edge":
        edge_float = edge_damage.astype(float) / 255.0
        result = stained * edge_float
    else:
        result = stained.copy()
        result[edge_damage <= 20] = 255

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


EFFECTS: Dict[str, Tuple[Callable[..., Image.Image], float, Callable[[], Dict]]] = {
    "rotate": (rotate, 0.2, lambda: {"angle": random.uniform(-3, 2)}),
    "bad_photocopy": (bad_photocopy, 0.05, lambda: {}),
    "lighting_gradient": (lighting_gradient, 0.025, lambda: {}),
    "page_border": (page_border, 0.025, lambda: {}),
    "folding": (folding, 0.05, lambda: {}),
    "shadow_cast": (shadow_cast, 0.0125, lambda: {}),
    "voronoi_tessellation": (voronoi_tessellation, 0.1, lambda: {}),
    # "reflected_light": (reflected_light, 0.0125, lambda: {}),
    "paper_stains_and_damage": (paper_stains_and_damage, 0.1, lambda: {}),
}


def build_effect_chain() -> List[Tuple[Callable[..., Image.Image], Dict]]:
    chain: List[Tuple[Callable[..., Image.Image], Dict]] = []
    for func, prob, param_gen in EFFECTS.values():
        if random.random() < prob:
            params = param_gen()
            chain.append((func, params))
    return chain


def apply_effects(img: Image.Image) -> Image.Image:
    # 10% chance to apply no effects
    if random.random() < NO_EFFECT:
        return img

    chain = build_effect_chain()
    for func, params in chain:
        img = func(img, **params)
    return img
