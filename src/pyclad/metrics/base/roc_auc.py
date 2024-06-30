from sklearn.metrics import roc_auc_score

from pyclad.metrics.base.base_metric import BaseMetric


class RocAuc(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return roc_auc_score(y_true=y_true, y_score=anomaly_scores)

    def name(self) -> str:
        return "ROC-AUC"
