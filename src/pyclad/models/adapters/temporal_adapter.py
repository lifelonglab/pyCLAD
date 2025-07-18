import numpy as np

from pyclad.models.model import Model


class FlattenTimeSeriesAdapter(Model):
    """
    Adapter to flatten time series data for models that expect 2D input.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, data: np.ndarray):
        data = data.reshape(data.shape[0], -1)
        self.model.fit(data)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = data.reshape(data.shape[0], -1)
        return self.model.predict(data)

    def name(self) -> str:
        return "FlattenedTS_" + self.model.name()
