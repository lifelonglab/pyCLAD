from typing import Optional

from sklearn.metrics import f1_score

from pyclad.vision.metrics.pixel_metric import PixelMetric
from pyclad.vision.metrics.pixel_threshold_utils import flatten_binary_masks


class PixelF1Score(PixelMetric):
    def __init__(self, threshold: Optional[float] = None, *, quantile: float = 0.99, zero_division: float = 0.0):
        super().__init__(threshold, quantile=quantile)
        self.zero_division = float(zero_division)

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_pred_flat = flatten_binary_masks(self._binarize(anomaly_scores))
        y_true_flat = flatten_binary_masks(y_true)
        return float(f1_score(y_true=y_true_flat, y_pred=y_pred_flat, zero_division=self.zero_division))

    def name(self) -> str:
        return "Pixel-F1-Score"
