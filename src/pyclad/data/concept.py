from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Concept:
    name: str
    data: np.array
    labels: Optional[np.array] = None
