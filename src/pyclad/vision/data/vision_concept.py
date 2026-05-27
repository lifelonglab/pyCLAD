from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyclad.data.concept import Concept


@dataclass
class VisionConcept(Concept):
    """A :class:`Concept` carrying image data with optional per-pixel anomaly masks.

    * ``data``: ``(N, H, W, C)`` array, or ``(N,)`` object array of image paths when ``data_mode="paths"``.
    * ``labels``: ``(N,)`` int64 with image-level anomaly labels (0 = normal, 1 = anomaly). ``None`` on training concepts.
    * ``masks``: ``(N, H, W)`` uint8 binary masks aligned with ``data`` on the batch dimension. ``None`` when no ground truth.
    """

    masks: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.masks is None:
            return
        if not hasattr(self.data, "shape"):
            return
        if self.masks.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"VisionConcept.masks must align with data on the batch dimension: "
                f"got masks.shape[0]={self.masks.shape[0]} vs data.shape[0]={self.data.shape[0]}"
            )
