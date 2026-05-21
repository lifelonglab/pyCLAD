from sklearn.metrics import f1_score

from pyclad.metrics.base.base_metric import BaseMetric


class F1Score(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)

    def name(self) -> str:
        return "F1-Score"
