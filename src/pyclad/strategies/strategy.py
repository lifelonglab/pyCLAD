import abc
from typing import Any, Dict

import numpy as np

from pyclad.output.output_writer import InfoProvider


class Strategy(InfoProvider):
    """Base class for all continual learning strategies."""

    @abc.abstractmethod
    def name(self) -> str: ...

    def additional_info(self) -> Dict:
        return {}

    def info(self) -> Dict[str, Any]:
        return {"strategy": {"name": self.name(), **self.additional_info()}}


class ConceptAwareStrategy(Strategy):

    @abc.abstractmethod
    def learn(self, data: np.ndarray, concept_id: str) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray, concept_id: str) -> (np.ndarray, np.ndarray): ...


class ConceptIncrementalStrategy(Strategy):
    @abc.abstractmethod
    def learn(self, data: np.ndarray) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray): ...


class ConceptAgnosticStrategy(Strategy):
    @abc.abstractmethod
    def learn(self, data: np.ndarray) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray): ...
