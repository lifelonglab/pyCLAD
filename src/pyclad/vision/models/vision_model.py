from abc import abstractmethod

import numpy as np

from pyclad.models.model import Model
from pyclad.vision.prediction_results import VisionPredictionResults


class VisionModel(Model):
    """Anomaly detection model whose predict() returns pixel-level score maps."""

    @abstractmethod
    def predict(self, data: np.ndarray) -> VisionPredictionResults: ...