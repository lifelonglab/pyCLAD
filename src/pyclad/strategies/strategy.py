import abc
from typing import Any, Dict, Optional

import numpy as np

from pyclad.models.model import Model
from pyclad.output.output_writer import InfoProvider


class Strategy(InfoProvider):
    """Base class for all continual learning strategies."""

    @abc.abstractmethod
    def name(self) -> str: ...

    def additional_info(self) -> Dict:
        return {}

    def info(self) -> Dict[str, Any]:
        return {"strategy": {"name": self.name(), **self.additional_info()}}

    def model_for_concept(self, concept_id: str) -> Optional[Model]:
        """Return the model handling ``concept_id`` so callbacks can inspect it.

        Default returns ``self._model`` if present, else ``None`` — covering
        every single-model strategy. Strategies that keep a per-concept model
        dict (e.g. :class:`pyclad.strategies.baselines.mste.MSTE`) MUST override.

        Callbacks needing typed access to model methods (e.g.
        :class:`pyclad.vision.callbacks.vision_pixel_concept_metric_callback.VisionPixelConceptMetricCallback`
        which needs ``score_maps``) call this instead of reflecting on
        strategy-private fields.
        """
        return getattr(self, "_model", None)


class ConceptAwareStrategy(Strategy):

    @abc.abstractmethod
    def learn(self, data: np.ndarray, concept_id: str) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray, concept_id: str) -> (np.ndarray, np.ndarray):
        """
        :param concept_id:
        :param data:
        :return: (predicted labels (0 for normal class, 1 for anomaly), anomaly scores (the higher the more anomalous))
        """
        ...


class ConceptIncrementalStrategy(Strategy):
    @abc.abstractmethod
    def learn(self, data: np.ndarray) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        :param data:
        :return: (predicted labels (0 for normal class, 1 for anomaly), anomaly scores (the higher the more anomalous))
        """
        ...


class ConceptAgnosticStrategy(Strategy):
    @abc.abstractmethod
    def learn(self, data: np.ndarray) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        :param data:
        :return: (predicted labels (0 for normal class, 1 for anomaly), anomaly scores (the higher the more anomalous))
        """
        ...
