import numpy as np

from pyclad.models.model import Model
from pyclad.output.prediction_results import PredictionResults


class MockModel(Model):

    def fit(self, data: np.ndarray):
        pass

    def predict(self, data: np.ndarray) -> PredictionResults:
        return PredictionResults(y_pred=np.array([]), anomaly_scores=np.array([]))

    def name(self) -> str:
        pass
