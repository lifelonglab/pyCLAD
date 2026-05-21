import numpy as np
from sklearn.metrics import average_precision_score

from pyclad.metrics.base.base_metric import BaseMetric


class AveragePrecision(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_true_arr = np.asarray(y_true).reshape(-1)
        y_score = np.asarray(anomaly_scores).reshape(-1)
        if len(np.unique(y_true_arr)) < 2:
            return float("nan")
        return float(average_precision_score(y_true=y_true_arr, y_score=y_score))

    def name(self) -> str:
        return "AP"
