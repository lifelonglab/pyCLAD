import abc
from abc import abstractmethod

import numpy as np


class Model(abc.ABC):
    @abstractmethod
    def learn(self, data: np.ndarray): ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param data:
        :return: (anomaly score, predicted labels)
        """
        ...
