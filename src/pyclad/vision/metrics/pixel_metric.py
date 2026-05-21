import abc

from pyclad.metrics.base.base_metric import BaseMetric


class PixelMetric(BaseMetric, abc.ABC):
    """Base class for pixel-level metrics that require a binarization threshold."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = float(threshold)
