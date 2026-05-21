import numpy as np

from pyclad.metrics.base.average_precision import AveragePrecision


def test_average_precision_perfect_ranking():
    metric = AveragePrecision()
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0])
    assert metric.compute(anomaly_scores=scores, y_pred=np.array([]), y_true=labels) == 1.0


def test_average_precision_single_class_returns_nan():
    metric = AveragePrecision()
    scores = np.array([0.5, 0.6, 0.7])
    labels = np.array([0, 0, 0])
    assert np.isnan(metric.compute(anomaly_scores=scores, y_pred=np.array([]), y_true=labels))


def test_average_precision_name():
    assert AveragePrecision().name() == "AP"
