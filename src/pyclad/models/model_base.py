import abc
from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from pyclad.output.output_writer import InfoProvider


class Model(InfoProvider):
    @abstractmethod
    def learn(self, data: np.ndarray): ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param data:
        :return: (anomaly score, predicted labels)
        """
        ...

    @abc.abstractmethod
    def name(self) -> str: ...

    def info(self) -> Dict[str, Any]:
        return {"model": {"name": self.name()}}
