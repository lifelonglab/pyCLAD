import abc

import numpy as np


class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def update(self, data: np.ndarray) -> None: ...

    @abc.abstractmethod
    def data(self) -> np.ndarray: ...

    @abc.abstractmethod
    def name(self) -> str: ...