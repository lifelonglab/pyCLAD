from dataclasses import dataclass
import numpy as np


@dataclass
class Concept:
    name: str
    data: np.array
    labels: np.array
