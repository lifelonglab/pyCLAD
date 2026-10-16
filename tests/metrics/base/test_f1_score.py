import numpy as np
import pytest

from pyclad.metrics.base.f1_score import F1Score


def test_f1_on_a_hand_computed_example():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 0])  # TP = 1, FP = 1, FN = 1

    assert F1Score().compute(anomaly_scores=np.array([]), y_pred=predictions, y_true=labels) == pytest.approx(0.5)


def test_f1_is_nan_without_anomalies_and_without_predicted_anomalies():
    zeros = np.zeros(4)

    assert np.isnan(F1Score().compute(anomaly_scores=np.array([]), y_pred=zeros, y_true=zeros))


@pytest.mark.parametrize(
    "labels, predictions",
    [
        (np.array([0, 0, 0, 0]), np.array([0, 1, 0, 0])),  # a false alarm on a normal-only concept
        (np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])),  # anomalies present, none predicted
    ],
)
def test_f1_is_zero_when_defined_but_nothing_is_right(labels, predictions):
    assert F1Score().compute(anomaly_scores=np.array([]), y_pred=predictions, y_true=labels) == 0.0
