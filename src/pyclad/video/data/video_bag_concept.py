"""Complete video bags as rows of a standard pyCLAD concept."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from pyclad.data.concept import Concept


@dataclass
class VideoBagConcept(Concept):
    """A concept with one VideoBag per row and optional frame labels.

    Keeping a bag in each object-array row lets the standard scenario preserve
    temporal order and replay identity without treating windows as samples.
    """

    frame_labels: Dict[str, np.ndarray] = field(default_factory=dict)
