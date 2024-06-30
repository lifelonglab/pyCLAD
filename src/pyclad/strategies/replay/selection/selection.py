import abc

import numpy as np


class SelectionMethod(abc.ABC):

    @abc.abstractmethod
    def select(self, data: np.ndarray, size: int) -> np.ndarray: ...
