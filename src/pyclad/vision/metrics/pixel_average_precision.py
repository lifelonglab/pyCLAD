import numpy as np
from sklearn.metrics import average_precision_score

from pyclad.metrics.base.base_metric import BaseMetric


class PixelAveragePrecision(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_score = np.asarray(anomaly_scores, dtype=np.float32).reshape(-1)
        y_true_flat = np.asarray(y_true, dtype=np.uint8).reshape(-1)
        if len(np.unique(y_true_flat)) < 2:
            return float("nan")
        return float(average_precision_score(y_true=y_true_flat, y_score=y_score))

    def name(self) -> str:
        return "Pixel-AP"
