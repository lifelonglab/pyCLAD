from unittest.mock import MagicMock

import numpy as np
import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)


class BaseMetricMock(BaseMetric):

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return 0

    def name(self) -> str:
        return "BaseMetricMock"


def _get_info(callback):
    return callback.info()["concept_metric_callback_BaseMetricMock"]


class ContinualMetricMock(ConceptLevelMetric):

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        pass

    def name(self) -> str:
        return "ContinualMetricMock"


def test_providing_info_without_evaluation():
    base_metric = BaseMetricMock()
    metrics = [ContinualMetricMock()]
    callback = ConceptMetricCallback(base_metric, metrics)

    assert _get_info(callback)["metrics"] == {m.name(): m.compute([[]]) for m in metrics}
    assert _get_info(callback)["concepts_order"] == []
    assert _get_info(callback)["metric_matrix"] == {}


@pytest.mark.parametrize(
    "concepts",
    [
        ["concept3", "concept1", "concept2"],
        ["concept12", "concept1", "concept5"],
        ["walking", "running", "jogging"],
        ["jogging", "running", "walking"],
    ],
)
def test_keeping_order_of_learned_concepts(concepts):
    base_metric = BaseMetricMock()
    metrics = [ContinualMetricMock()]
    callback = ConceptMetricCallback(base_metric, metrics)

    for concept in concepts:
        callback.after_training(Concept(concept, data=np.array([]), labels=np.array([])))

        for evaluation_concept in concepts:
            callback.after_evaluation(
                Concept(evaluation_concept, data=np.array([]), labels=np.array([])),
                np.array([]),
                np.array([]),
                np.array([]),
            )

    assert _get_info(callback)["concepts_order"] == concepts


def test_computing_metrics():
    base_metric = BaseMetricMock()
    metric = ContinualMetricMock()
    metric.compute = MagicMock(return_value=1.0)
    metrics = [metric]
    callback = ConceptMetricCallback(base_metric, metrics)

    for metric in metrics:
        assert metric.compute([[]]) == _get_info(callback)["metrics"][metric.name()]
