from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyclad.data.concept import Concept


@dataclass
class VisionConcept(Concept):
    masks: Optional[np.ndarray] = None
