import numpy as np
from sklearn.metrics import roc_auc_score

from pyclad.metrics.base.base_metric import BaseMetric


class RocAuc(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        # Undefined with a single class. Checked here so the answer does not depend on the scikit-learn version.
        if len(np.unique(np.asarray(y_true))) < 2:
            return float("nan")
        return float(roc_auc_score(y_true=y_true, y_score=anomaly_scores))

    def name(self) -> str:
        return "ROC-AUC"
