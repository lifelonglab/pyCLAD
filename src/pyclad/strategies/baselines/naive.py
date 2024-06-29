import numpy as np

from pyclad.models.model_base import Model
from pyclad.strategies.strategy import ConceptIncrementalStrategy, ConceptAwareStrategy


class NaiveStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):

    def __init__(self, model: Model):
        self._model = model

    def learn(self, data: np.ndarray) -> None:
        self._model.learn(data)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return 'Naive'
