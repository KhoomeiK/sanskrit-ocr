from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional
from PIL import Image

from .render_text import generate_sanskrit_samples
from .render_book_page import generate_book_pages
from .render_parchment_leaf import generate_parchment_leaves

# Registry for available renderers
RENDERERS: Dict[str, Callable[[str, Optional[str], Optional[dict]], List[Image.Image]]] = {
    "text": generate_sanskrit_samples,
    "book": generate_book_pages,
    "parchment": generate_parchment_leaves,
}


def register_renderer(name: str, func: Callable[[str, Optional[str], Optional[dict]], List[Image.Image]]) -> None:
    """Register a new rendering function."""
    RENDERERS[name] = func


def render_image(
    text: str,
    *,
    renderer: str = "random",
    output_dir: Optional[str] = None,
    params: Optional[dict] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Image.Image]:
    """Render text into images using one of the registered renderers.

    Args:
        text: Text to render.
        renderer: Name of the renderer to use. Set to ``"random"`` to pick
            one at random based on ``weights``.
        output_dir: Optional directory to save generated images.
        params: Parameters passed through to the renderer.
        weights: Optional weights for random renderer selection.

    Returns:
        List of :class:`PIL.Image.Image` objects or ``None`` if ``output_dir``
        is provided and the renderer saves images directly.
    """
    if renderer == "random":
        names = list(RENDERERS)
        wts = [weights.get(n, 1.0) if weights else 1.0 for n in names]
        renderer = random.choices(names, weights=wts, k=1)[0]

    if renderer not in RENDERERS:
        raise ValueError(f"Unknown renderer '{renderer}'")

    func = RENDERERS[renderer]
    return func(text, output_dir=output_dir, params=params)
