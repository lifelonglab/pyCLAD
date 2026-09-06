from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConceptNode:
    """A single concept (or sub-concept) in VLAD's hierarchical memory; a `C_(l,l')` node in the
    paper. `layer` is the hierarchy depth (1 = root); `parent_id` is `None` for root concepts.
    `threshold` is the concept's Wasserstein admission threshold (`E[D]`, Eq. 2) — `None` until
    enough samples have accumulated to calibrate it ("still forming").
    """

    id: str
    parent_id: Optional[str]
    layer: int
    samples: np.ndarray
    threshold: Optional[float] = None
