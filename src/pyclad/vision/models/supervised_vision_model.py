from abc import abstractmethod
from typing import Optional

import numpy as np

from pyclad.models.supervised_model import SupervisedModel
from pyclad.vision.prediction_results import VisionPredictionResults


class SupervisedVisionModel(SupervisedModel):
    @abstractmethod
    def fit(self, data: np.ndarray, labels: np.ndarray, masks: Optional[np.ndarray] = None) -> None: ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> VisionPredictionResults: ...
