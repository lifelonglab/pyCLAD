import numpy as np
from sklearn.metrics import f1_score

from pyclad.metrics.base.base_metric import BaseMetric


class F1Score(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        # Undefined when there is no anomaly to find and none was predicted (2TP + FP + FN = 0). Checked here so the
        # answer does not depend on scikit-learn's zero_division default.
        if not np.any(np.asarray(y_true) == 1) and not np.any(np.asarray(y_pred) == 1):
            return float("nan")
        return float(f1_score(y_true=y_true, y_pred=y_pred))

    def name(self) -> str:
        return "F1-Score"
