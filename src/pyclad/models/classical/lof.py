import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from pyclad.models.classical.utils import adjust_scikit_predictions
from pyclad.models.model_base import Model


class LocalOutlierFactorAdapter(Model):
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)

    def learn(self, data: np.ndarray):
        self.model.fit(data)

    def predict(self, data: np.ndarray):
        return adjust_scikit_predictions(self.model.predict(data)), -self.model.score_samples(data)

    def name(self) -> str:
        return "LocalOutlierFactor"

    def additional_info(self):
        return {"n_neighbors": self.n_neighbors}
