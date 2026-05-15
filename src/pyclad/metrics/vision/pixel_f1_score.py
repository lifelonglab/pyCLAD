from sklearn.metrics import f1_score

from pyclad.metrics.vision.pixel_metric import PixelMetric
from pyclad.metrics.vision.pixel_threshold_utils import (
    flatten_binary_masks,
    threshold_pixel_scores,
)


class PixelF1Score(PixelMetric):
    def __init__(self, threshold: float = 0.5, zero_division: float = 0.0):
        super().__init__(threshold)
        self.zero_division = float(zero_division)

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_pred_flat = flatten_binary_masks(threshold_pixel_scores(anomaly_scores, self.active_threshold()))
        y_true_flat = flatten_binary_masks(y_true)
        return float(f1_score(y_true=y_true_flat, y_pred=y_pred_flat, zero_division=self.zero_division))

    def name(self) -> str:
        return "Pixel-F1-Score"
