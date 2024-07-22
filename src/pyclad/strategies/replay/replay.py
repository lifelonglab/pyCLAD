import numpy as np

from pyclad.models.model_base import Model
from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class ReplayOnlyStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    def __init__(self, model: Model, buffer: ReplayBuffer):
        self._model = model
        self._buffer = buffer

    def learn(self, data: np.ndarray) -> None:
        self._buffer.update(data)
        self._model.learn(self._buffer.data())

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "ReplayOnly"


class ReplayEnhancedStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    def __init__(self, model: Model, buffer: ReplayBuffer):
        self._model = model
        self._buffer = buffer

    def learn(self, data: np.ndarray) -> None:
        self._model.learn(np.concatenate([self._buffer.data(), data]))
        self._buffer.update(data)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "ReplayEnhanced"
