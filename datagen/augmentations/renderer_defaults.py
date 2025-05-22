# Defaults for augmentation renderers
# Extracted to centralize configuration

from __future__ import annotations

import random

# Utility to merge override dictionaries with defaults

def merge_params(defaults: dict, overrides: dict | None) -> dict:
    params = defaults.copy()
    if overrides:
        params.update(overrides)
    return params

# --- render_text defaults ---
TEXT_DEFAULTS = {
    'width': 400,
    'height': 320,
    'base_images': 1,
    'font_dir': '/home/ubuntu/sanskrit-ocr/datagen/fonts',
    'font': 'Sharad76-Regular.otf',
    'noise': 0.7,
    'aging': 0.6,
    'texture': 0.7,
    'stains': 0.6,
    'stain_intensity': 0.5,
    'word_position': 0.6,
    'ink_color': 0.5,
    'line_spacing': 0.4,
    'baseline': 0.3,
    'word_angle': 0.0,
    'apply_transforms': True,
    'all_transforms': False,
    'rotation_max': 5.0,
    'brightness_var': 0.2,
    'contrast_var': 0.2,
    'noise_min': 0.01,
    'noise_max': 0.05,
    'blur_min': 0.5,
    'blur_max': 1.0,
}

# --- render_book_page defaults ---
size = 2
BOOK_PAGE_DEFAULTS = {
    'output_dir': 'output',
    'image_dpi': 300,
    'no_degrade_prob': 0.1,
    'layouts': ["vanilla", "columns", "footnote", "subheading"],
    'layout_weights': [0.25, 0.25, 0.25, 0.25],
    'effects': {
        "blur": (0.3, lambda: dict(radius=random.choice([3, 5, 7]))),
        "bleed_through": (0.15, lambda: dict(
            alpha=round(random.uniform(0.7, 0.9), 2),
            offset_y=random.randint(-10, 10)
        )),
        "salt": (0.2, lambda: dict(amount=round(random.uniform(0.01, 0.05), 3))),
        "pepper": (0.2, lambda: dict(amount=0.03)),
        "morphology": (0.75, lambda: (
            lambda kernel_type: dict(
                operation=random.choices(
                    ["open", "close", "dilate", "erode"],
                    weights=[1, 1, 1, 2]
                )[0],
                kernel_type=kernel_type,
                kernel_shape=(
                    (1, size) if random.choice([True, False]) else (size, 1)
                ) if kernel_type == "ones" else (size, size)
            )
        )(kernel_type := random.choice(["ones", "upper_triangle", "lower_triangle", "x", "plus", "ellipse"])))
    }
}

# --- render_parchment_leaf defaults ---
PARCHMENT_DEFAULTS = {
    'output_dir': 'output_parchment',
    'dpi': 300,
    'no_degrade_prob': 0.10,
    'parchment_colors': ["#f5deb3", "#f0d8ab", "#c3a374", "#a47a3c"],
    'ink_colors': ["#100d05", "#23140a"],
    'effects': {
        "blur": (0.3, lambda: dict(radius=random.choice([3, 5, 7]))),
        "bleed_through": (0.15, lambda: dict(
            alpha=random.uniform(0.7, 0.9),
            offset_y=random.randint(-8, 8)
        )),
        "salt": (0.2, lambda: dict(amount=random.uniform(0.03, 0.06))),
        "pepper": (0.2, lambda: dict(amount=0.03)),
        "morphology": (0.75, lambda: dict(
            operation=random.choices(
                ["open", "close", "dilate", "erode"],
                weights=[1, 1, 1, 2]
            )[0],
            kernel_type=random.choice([
                "ones", "upper_triangle", "lower_triangle",
                "x", "plus", "ellipse"
            ]),
            kernel_shape=(
                (1, size) if random.choice([True, False]) else (size, 1)
            ) if random.choice([True, False]) else (size, size)
        ))
    }
}
