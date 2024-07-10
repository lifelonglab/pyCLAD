import numpy as np
from sklearn.ensemble import IsolationForest

from pyclad.models.adapters.utils import adjust_predictions
from pyclad.models.model_base import Model


class IsolationForestAdapter(Model):
    def __init__(self, n_estimators=100, contamination=0.00001):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=0.00001)

    def learn(self, data: np.ndarray):
        self.model.fit(data)

    def predict(self, data: np.ndarray):
        return adjust_predictions(self.model.predict(data)), -self.model.score_samples(data)

    def name(self) -> str:
        return "IsolationForest"
