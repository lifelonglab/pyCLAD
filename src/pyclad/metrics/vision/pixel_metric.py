import abc

from pyclad.metrics.base.base_metric import BaseMetric


class PixelMetric(BaseMetric, abc.ABC):
    """Base class for pixel-level metrics that require a binarization threshold."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = float(threshold)
        self._runtime_threshold: float | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_runtime_threshold(self, threshold: float | None) -> None:
        self._runtime_threshold = None if threshold is None else float(threshold)

    def active_threshold(self) -> float:
        return self._threshold if self._runtime_threshold is None else self._runtime_threshold
