"""Pixel-level concept-metric callback under a step schedule (rectangular T x N matrix)."""

import numpy as np
import pytest

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.grouping import apply_step_schedule
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
)
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    ScheduleAwareVisionPixelConceptMetricCallback,
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.vision_concept import VisionConcept

TEST_CATEGORIES = ["bottle", "cable", "capsule"]
TRAIN_STEPS = ["step_0", "step_1"]
FIRST_SEEN = {"bottle": 0, "cable": 0, "capsule": 1}


class ScoreDrivenPixelMetric(BaseMetric):
    """Returns the constant the score map is filled with, so tests can pin matrix cells."""

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return float(np.asarray(anomaly_scores).ravel()[0])

    def name(self) -> str:
        return "ScoreDrivenPixel"


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


def _concept(name: str) -> VisionConcept:
    return VisionConcept(
        name=name,
        data=np.zeros((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
        masks=np.zeros((2, 2, 2), dtype=np.uint8),
    )


def _info(callback):
    return callback.info()["pixel_concept_metric_callback_ScoreDrivenPixel"]


def _run(callback, cell_values, test_categories=TEST_CATEGORIES, train_steps=TRAIN_STEPS):
    for step_index, step_name in enumerate(train_steps):
        callback.after_training(Concept(name=step_name, data=np.array([])))
        for category_index, category in enumerate(test_categories):
            concept = _concept(category)
            callback.after_evaluation(
                evaluated_concept=concept,
                y_true=concept.labels,
                y_pred=np.array([0, 0], dtype=np.int64),
                anomaly_scores=np.array([0.5, 0.5], dtype=np.float32),
                score_maps=np.full((2, 2, 2), cell_values[step_index][category_index], dtype=np.float32),
            )


def test_building_a_rectangular_matrix_from_grouped_training_steps():
    callback = VisionPixelConceptMetricCallback(
        base_metric=ScoreDrivenPixelMetric(), summarized_metrics=[FinalStepAverage()]
    )

    _run(callback, [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]])

    assert _info(callback)["metrics"]["FinalStepAverage"] == pytest.approx((0.6 + 0.4 + 0.7) / 3, abs=1e-6)


def test_reporting_train_and_test_order_separately():
    callback = VisionPixelConceptMetricCallback(base_metric=ScoreDrivenPixelMetric())

    _run(callback, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    assert _info(callback)["concepts_order"] == TRAIN_STEPS
    assert _info(callback)["test_order"] == TEST_CATEGORIES


def test_passing_the_aligned_first_seen_steps_to_schedule_aware_metrics():
    metric = RecordingScheduleAwareMetric()
    callback = ScheduleAwareVisionPixelConceptMetricCallback(
        base_metric=ScoreDrivenPixelMetric(),
        schedule_aware_metrics=[metric],
        first_seen_step=FIRST_SEEN,
    )

    _run(callback, [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]])
    info = _info(callback)

    assert metric.received_first_seen == [0, 0, 1]
    assert info["schedule_aware_metrics"]["RecordingScheduleAware"] == 0.42


def test_rejecting_schedule_aware_metrics_without_a_first_seen_mapping():
    with pytest.raises(ValueError, match="first_seen_step"):
        ScheduleAwareVisionPixelConceptMetricCallback(
            base_metric=ScoreDrivenPixelMetric(),
            schedule_aware_metrics=[RecordingScheduleAwareMetric()],
        )


def test_rejecting_a_matrix_with_a_missing_cell():
    callback = VisionPixelConceptMetricCallback(base_metric=ScoreDrivenPixelMetric())
    _run(callback, [[0.1, 0.2, 0.3]], train_steps=["step_0"])
    # A second step during which the strategy produced no score maps at all.
    callback.after_training(Concept(name="step_1", data=np.array([])))

    with pytest.raises(ValueError, match="was not evaluated"):
        callback.info()


def test_reporting_nothing_when_no_pixel_evaluation_happened():
    callback = VisionPixelConceptMetricCallback(base_metric=ScoreDrivenPixelMetric())
    callback.after_training(Concept(name="step_0", data=np.array([])))

    assert callback.info() == {}


def test_accepting_a_step_scheduled_dataset_instead_of_a_mapping():
    """Same wiring as the image-level callback, since both go through ScheduleAwareSupport."""
    dataset = ConceptsDataset(
        name="toy",
        train_concepts=[Concept(name=c, data=np.zeros((2, 1))) for c in TEST_CATEGORIES],
        test_concepts=[Concept(name=c, data=np.zeros((2, 1)), labels=np.zeros(2)) for c in TEST_CATEGORIES],
    )
    scheduled = apply_step_schedule(dataset, "2-1")
    metric = RecordingScheduleAwareMetric()
    callback = ScheduleAwareVisionPixelConceptMetricCallback(
        base_metric=ScoreDrivenPixelMetric(),
        schedule_aware_metrics=[metric],
        first_seen_step=scheduled,
    )

    _run(callback, [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]])
    info = _info(callback)

    assert metric.received_first_seen == [0, 0, 1]
    assert info["first_seen_step"] == {"bottle": 0, "cable": 0, "capsule": 1}
