import abc
from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from pyclad.output.output_writer import InfoProvider


class Model(InfoProvider):
    @abstractmethod
    def fit(self, data: np.ndarray): ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        :param data:
        :return: (predicted labels (0 for normal class, 1 for anomaly), anomaly scores (the higher the more anomalous))
        """
        ...

    @abc.abstractmethod
    def name(self) -> str: ...

    def info(self) -> Dict[str, Any]:
        return {"model": {"name": self.name(), **self.additional_info()}}

    def additional_info(self):
        return {}
