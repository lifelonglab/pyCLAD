import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class NaiveStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):

    def __init__(self, model: Model):
        self._model = model

    def learn(self, data: np.ndarray, **kwargs) -> None:
        self._model.fit(data)

    def predict(self, data: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "Naive"
