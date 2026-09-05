"""Concept-metric callback behaviour when training steps group several categories.

With a step schedule the train axis holds ``T`` grouped steps (``step_0``, ``step_1``, ...)
while the test axis keeps the ``N`` original category names, so the two axes no longer
share names and the matrix is rectangular.
"""

import numpy as np
import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import (
    ConceptMetricCallback,
    ScheduleAwareConceptMetricCallback,
    build_dense_matrix,
)
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.grouping import apply_step_schedule
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
)
from pyclad.metrics.continual.final_step_average import FinalStepAverage

TEST_CATEGORIES = ["bottle", "cable", "capsule"]
TRAIN_STEPS = ["step_0", "step_1"]
FIRST_SEEN = {"bottle": 0, "cable": 0, "capsule": 1}


class ScoreDrivenBaseMetric(BaseMetric):
    """Returns the single value passed in as ``anomaly_scores``, so tests can pin cells."""

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return float(anomaly_scores[0])

    def name(self) -> str:
        return "ScoreDriven"


class RecordingScheduleAwareMetric(ScheduleAwareMetric):
    def __init__(self):
        self.received_matrix = None
        self.received_first_seen = None

    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps) -> float:
        self.received_matrix = metric_matrix
        self.received_first_seen = list(first_seen_steps)
        return 0.42

    def name(self) -> str:
        return "RecordingScheduleAware"


def _info(callback):
    return callback.info()["concept_metric_callback_ScoreDriven"]


def _run(callback, cell_values, test_categories=TEST_CATEGORIES, train_steps=TRAIN_STEPS):
    """Drive the callback the way ConceptAwareScenario would, pinning each matrix cell."""
    for step_index, step_name in enumerate(train_steps):
        callback.after_training(Concept(name=step_name, data=np.array([])))
        for category_index, category in enumerate(test_categories):
            callback.after_evaluation(
                evaluated_concept=Concept(name=category, data=np.array([])),
                y_true=np.array([0]),
                y_pred=np.array([0]),
                anomaly_scores=np.array([cell_values[step_index][category_index]]),
            )


def test_building_a_rectangular_matrix_from_grouped_training_steps():
    callback = ConceptMetricCallback(ScoreDrivenBaseMetric(), summarized_metrics=[FinalStepAverage()])

    _run(callback, [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]])

    assert _info(callback)["metrics"]["FinalStepAverage"] == pytest.approx((0.6 + 0.4 + 0.7) / 3)


def test_reporting_train_and_test_order_separately():
    callback = ConceptMetricCallback(ScoreDrivenBaseMetric(), summarized_metrics=[])

    _run(callback, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    assert _info(callback)["concepts_order"] == TRAIN_STEPS
    assert _info(callback)["test_order"] == TEST_CATEGORIES


def test_rejecting_a_matrix_with_a_missing_cell():
    callback = ConceptMetricCallback(ScoreDrivenBaseMetric(), summarized_metrics=[])
    callback.after_training(Concept(name="step_0", data=np.array([])))
    for category in TEST_CATEGORIES:
        callback.after_evaluation(
            evaluated_concept=Concept(name=category, data=np.array([])),
            y_true=np.array([0]),
            y_pred=np.array([0]),
            anomaly_scores=np.array([0.5]),
        )
    # Second step evaluates only one of the three categories.
    callback.after_training(Concept(name="step_1", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=Concept(name="bottle", data=np.array([])),
        y_true=np.array([0]),
        y_pred=np.array([0]),
        anomaly_scores=np.array([0.5]),
    )

    with pytest.raises(ValueError, match="was not evaluated"):
        callback.info()


def test_passing_the_aligned_first_seen_steps_to_schedule_aware_metrics():
    metric = RecordingScheduleAwareMetric()
    callback = ScheduleAwareConceptMetricCallback(
        ScoreDrivenBaseMetric(),
        summarized_metrics=[],
        schedule_aware_metrics=[metric],
        first_seen_step=FIRST_SEEN,
    )

    _run(callback, [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]])
    info = _info(callback)

    assert metric.received_first_seen == [0, 0, 1]
    assert metric.received_matrix == [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]]
    assert info["schedule_aware_metrics"]["RecordingScheduleAware"] == 0.42
    assert info["first_seen_step"] == FIRST_SEEN


def test_realigning_first_seen_steps_to_the_evaluation_order():
    metric = RecordingScheduleAwareMetric()
    callback = ScheduleAwareConceptMetricCallback(
        ScoreDrivenBaseMetric(),
        summarized_metrics=[],
        schedule_aware_metrics=[metric],
        first_seen_step=FIRST_SEEN,
    )

    _run(callback, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], test_categories=["capsule", "bottle", "cable"])
    _info(callback)

    assert metric.received_first_seen == [1, 0, 0]


def test_rejecting_schedule_aware_metrics_without_a_first_seen_mapping():
    with pytest.raises(ValueError, match="first_seen_step"):
        ScheduleAwareConceptMetricCallback(
            ScoreDrivenBaseMetric(),
            summarized_metrics=[],
            schedule_aware_metrics=[RecordingScheduleAwareMetric()],
        )


def test_rejecting_a_first_seen_mapping_that_misses_an_evaluated_category():
    callback = ScheduleAwareConceptMetricCallback(
        ScoreDrivenBaseMetric(),
        summarized_metrics=[],
        schedule_aware_metrics=[RecordingScheduleAwareMetric()],
        first_seen_step={"bottle": 0, "cable": 0},
    )

    _run(callback, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with pytest.raises(ValueError, match="capsule"):
        callback.info()


def test_square_scenario_keeps_reporting_identical_train_and_test_order():
    callback = ConceptMetricCallback(ScoreDrivenBaseMetric(), summarized_metrics=[])

    _run(callback, [[0.1, 0.2], [0.3, 0.4]], test_categories=["a", "b"], train_steps=["a", "b"])

    assert _info(callback)["concepts_order"] == ["a", "b"]
    assert _info(callback)["test_order"] == ["a", "b"]


class TestAxisAlignment:
    """A square matrix must stay name-consistent: cell [i][i] means "concept i after learning i".

    Rows follow the training order and columns the evaluation order, and those are tracked
    independently. When both axes hold the same concepts, pairing them by position rather than
    by name silently transposes cells, and the square-only metrics then average the wrong ones
    without raising.
    """

    def test_columns_follow_the_training_order_when_both_axes_hold_the_same_concepts(self):
        metric_matrix = {"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.2, "B": 0.8}}

        matrix = build_dense_matrix(metric_matrix, train_order=["A", "B"], test_order=["B", "A"])

        assert matrix == [[0.9, 0.1], [0.2, 0.8]]

    def test_columns_keep_the_evaluation_order_when_the_axes_differ(self):
        metric_matrix = {"step_0": {"b": 0.1, "a": 0.2}, "step_1": {"b": 0.3, "a": 0.4}}

        matrix = build_dense_matrix(metric_matrix, train_order=["step_0", "step_1"], test_order=["b", "a"])

        assert matrix == [[0.1, 0.2], [0.3, 0.4]]

    def test_callback_reports_the_column_order_it_actually_used(self):
        callback = ConceptMetricCallback(ScoreDrivenBaseMetric(), summarized_metrics=[])
        _run(callback, [[0.1, 0.2], [0.3, 0.4]], test_categories=["b", "a"], train_steps=["a", "b"])

        info = _info(callback)

        assert info["concepts_order"] == ["a", "b"]
        assert info["test_order"] == ["a", "b"]


class TestScheduleAwareWiring:
    def test_reading_the_mapping_off_a_step_scheduled_dataset(self):
        dataset = ConceptsDataset(
            name="toy",
            train_concepts=[Concept(name=c, data=np.zeros((2, 1))) for c in TEST_CATEGORIES],
            test_concepts=[Concept(name=c, data=np.zeros((2, 1)), labels=np.zeros(2)) for c in TEST_CATEGORIES],
        )
        scheduled = apply_step_schedule(dataset, "2-1")
        metric = RecordingScheduleAwareMetric()
        callback = ScheduleAwareConceptMetricCallback(
            ScoreDrivenBaseMetric(),
            summarized_metrics=[],
            schedule_aware_metrics=[metric],
            first_seen_step=scheduled,
        )

        _run(callback, [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]])
        _info(callback)

        assert metric.received_first_seen == [0, 0, 1]

    def test_rejecting_schedule_aware_callback_without_any_schedule_aware_metric(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ScheduleAwareConceptMetricCallback(
                ScoreDrivenBaseMetric(), summarized_metrics=[], first_seen_step=FIRST_SEEN
            )

    def test_plain_callback_no_longer_accepts_schedule_aware_arguments(self):
        with pytest.raises(TypeError):
            ConceptMetricCallback(ScoreDrivenBaseMetric(), summarized_metrics=[], first_seen_step=FIRST_SEEN)
