import numpy as np

from pyclad.models.model import Model


class MockModel(Model):

    def learn(self, data: np.ndarray):
        pass

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        pass

    def name(self) -> str:
        pass
