"""Shared writers for synthetic image / mask fixtures across vision tests.

Centralizes what every loader/reader/HF test needs to populate ``tmp_path``
before invoking pyclad APIs. Sizes default to 4×4 (smallest valid PIL image
batch); pass ``size=(H, W)`` when the test needs a specific shape.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def write_rgb_image(
    path: Path,
    color: tuple[int, int, int] = (128, 64, 32),
    size: tuple[int, int] = (4, 4),
) -> None:
    """Write a solid-color RGB image at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def write_mask(path: Path, value: int = 255, size: tuple[int, int] = (4, 4)) -> None:
    """Write a single-channel uint8 mask at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size, value, dtype=np.uint8), mode="L").save(path)
