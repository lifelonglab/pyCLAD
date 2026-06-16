import abc
from typing import Optional

import numpy as np

from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.vision.metrics.pixel_threshold_utils import (
    resolve_pixel_threshold,
    threshold_pixel_scores,
)


class PixelMetric(BaseMetric, abc.ABC):
    """Base class for pixel-level metrics that binarize per-pixel anomaly scores.

    The binarization threshold is either fixed or derived from the score
    distribution. Anomaly-map scales differ wildly across models — e.g. PaSTe
    distance maps are unbounded and positive, while FastFlow scores live in
    ``[-1, 0)`` — so a single fixed threshold cannot serve all of them: a value
    tuned for one model silently collapses the metric to 0 for another. By
    default the threshold is therefore data-driven (a high quantile of the
    evaluated score map), which is scale-invariant. Passing an explicit
    ``threshold`` opts back into fixed binarization.

    Caveat: the quantile default trades an arbitrary *value* for an arbitrary
    *anomalous-pixel fraction* (``1 - quantile``, ~1% at the default). On
    benchmarks with large defects this biases F1/Dice/IoU downward. For a
    threshold-free assessment of localization quality prefer rank-based metrics
    (Pixel-ROC-AUC, Pixel-AP, Pixel-AUPRO), which do not need binarization.
    """

    def __init__(self, threshold: Optional[float] = None, *, quantile: float = 0.99):
        self._fixed_threshold: Optional[float] = None if threshold is None else float(threshold)
        self._quantile = float(quantile)

    def _binarize(self, anomaly_scores) -> np.ndarray:
        threshold = resolve_pixel_threshold(
            anomaly_scores,
            mode="fixed" if self._fixed_threshold is not None else "train-quantile",
            fixed_threshold=self._fixed_threshold if self._fixed_threshold is not None else 0.5,
            quantile=self._quantile,
        )
        return threshold_pixel_scores(anomaly_scores, threshold)
