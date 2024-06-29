import random

import numpy as np

from pyclad.strategies.replay.selection.selection import SelectionMethod


class RandomSelection(SelectionMethod):
    def select(self, data: np.ndarray, size: int) -> np.ndarray:
        selected_indices = random.sample(range(len(data)), size)
        return data[selected_indices]
