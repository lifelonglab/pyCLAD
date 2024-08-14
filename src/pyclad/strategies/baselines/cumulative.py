from typing import Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptAwareStrategy,
    ConceptIncrementalStrategy,
)


class CumulativeStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    def __init__(self, model: Model):
        self._replay = []
        self._model = model

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        """Learn from the data and store it in the replay buffer."""
        self._replay.append(data)
        self._model.fit(np.concatenate(self._replay))

    def predict(self, data: np.ndarray, *args, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "Cumulative"

    def additional_info(self) -> Dict:
        return {"model": self._model.name(), "buffer_size": len(np.concatenate(self._replay))}
