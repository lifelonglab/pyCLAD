import numpy as np

from pyclad.models.model_base import Model
from pyclad.strategies.strategy import ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy


class CumulativeStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    def __init__(self, model: Model):
        self._replay = []
        self._model = model

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        self._replay.append(data)
        self._model.learn(np.concatenate(self._replay))

    def predict(self, data: np.ndarray, *args, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return 'Cumulative'
