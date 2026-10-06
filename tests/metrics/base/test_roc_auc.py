import numpy as np
import pytest

from pyclad.metrics.base.roc_auc import RocAuc


def test_roc_auc_ranks_scores_against_labels():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    assert RocAuc().compute(anomaly_scores=scores, y_pred=np.array([]), y_true=labels) == pytest.approx(0.75)


@pytest.mark.parametrize("labels", [np.zeros(4), np.ones(4)])
def test_roc_auc_single_class_returns_nan(labels):
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    assert np.isnan(RocAuc().compute(anomaly_scores=scores, y_pred=np.array([]), y_true=labels))
