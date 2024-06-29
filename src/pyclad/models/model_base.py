import abc
import numpy as np
from abc import abstractmethod


class Model(abc.ABC):
    @abstractmethod
    def learn(self, data: np.ndarray):
        ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param data:
        :return: (anomaly score, predicted labels)
        """
        ...
