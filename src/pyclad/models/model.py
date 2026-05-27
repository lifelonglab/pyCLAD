import abc
from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from pyclad.output.output_writer import InfoProvider
from pyclad.output.prediction_results import PredictionResults


class Model(InfoProvider):
    @abstractmethod
    def fit(self, data: np.ndarray): ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> PredictionResults: ...

    @abc.abstractmethod
    def name(self) -> str: ...

    def info(self) -> Dict[str, Any]:
        return {"model": {"name": self.name(), **self.additional_info()}}

    def additional_info(self):
        return {}
