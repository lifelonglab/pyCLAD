"""End-to-end check that a step-scheduled dataset runs through a scenario and reports metrics.

Covers the seam between the three layers that grouping touches: the grouped dataset feeds
the unchanged scenario, the scenario drives the callback with train-step names on one axis
and category names on the other, and the schedule-aware metrics land in the output payload.
"""

import numpy as np
import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import (
    ConceptMetricCallback,
    ScheduleAwareConceptMetricCallback,
)
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.grouping import apply_step_schedule
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure
from pyclad.metrics.continual.schedule_aware_forgetting_measure import (
    ScheduleAwareForgettingMeasure,
)
from pyclad.metrics.continual.schedule_aware_forward_transfer import (
    ScheduleAwareForwardTransfer,
)
from pyclad.metrics.continual.schedule_aware_new_task_acquisition import (
    ScheduleAwareNewTaskAcquisition,
)
from pyclad.models.model import Model
from pyclad.output.prediction_results import PredictionResults
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.naive import NaiveStrategy

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid"]


class ConstantScoreModel(Model):
    def fit(self, data: np.ndarray) -> None:
        pass

    def predict(self, data: np.ndarray) -> PredictionResults:
        rows = len(data)
        return PredictionResults(
            y_pred=np.zeros(rows, dtype=np.int64),
            anomaly_scores=np.linspace(0.0, 1.0, num=rows),
        )

    def name(self) -> str:
        return "ConstantScoreModel"


class ConstantBaseMetric(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return 0.75

    def name(self) -> str:
        return "Constant"


@pytest.fixture
def dataset():
    return ConceptsDataset(
        name="toy",
        train_concepts=[Concept(name=category, data=np.zeros((4, 3), dtype=np.float64)) for category in CATEGORIES],
        test_concepts=[
            Concept(
                name=category,
                data=np.zeros((4, 3), dtype=np.float64),
                labels=np.array([0, 0, 1, 1], dtype=np.int64),
            )
            for category in CATEGORIES
        ],
    )


def _run_scenario(dataset, callbacks):
    ConceptAwareScenario(dataset=dataset, strategy=NaiveStrategy(ConstantScoreModel()), callbacks=callbacks).run()


def test_step_scheduled_run_reports_a_rectangular_matrix(dataset):
    scheduled = apply_step_schedule(dataset, "3-1-1")
    callback = ScheduleAwareConceptMetricCallback(
        base_metric=ConstantBaseMetric(),
        summarized_metrics=[FinalStepAverage()],
        schedule_aware_metrics=[
            ScheduleAwareForgettingMeasure(),
            ScheduleAwareForwardTransfer(),
            ScheduleAwareNewTaskAcquisition(),
        ],
        first_seen_step=scheduled.first_seen_step(),
    )

    _run_scenario(scheduled, [callback])
    info = callback.info()["concept_metric_callback_Constant"]

    assert info["concepts_order"] == ["step_0", "step_1", "step_2"]
    assert info["test_order"] == CATEGORIES
    assert info["first_seen_step"] == {"bottle": 0, "cable": 0, "capsule": 0, "carpet": 1, "grid": 2}
    assert info["metrics"]["FinalStepAverage"] == pytest.approx(0.75)
    assert set(info["schedule_aware_metrics"]) == {
        "ScheduleAwareForgettingMeasure",
        "ScheduleAwareForwardTransfer",
        "ScheduleAwareNewTaskAcquisition",
    }


def test_dataset_info_carries_the_schedule(dataset):
    scheduled = apply_step_schedule(dataset, "3-1-1")

    info = scheduled.info()["dataset"]

    assert info["name"] == "toy__3-1x2"
    assert info["step_schedule_parsed"] == [3, 1, 1]
    assert info["train_steps"] == ["step_0", "step_1", "step_2"]


def test_square_metrics_reject_the_step_scheduled_matrix(dataset):
    """A square-only metric must fail loudly rather than average meaningless cells."""
    scheduled = apply_step_schedule(dataset, "3-1-1")
    callback = ConceptMetricCallback(
        base_metric=ConstantBaseMetric(),
        summarized_metrics=[],
        stepwise_metrics=[ForgettingMeasure()],
    )

    _run_scenario(scheduled, [callback])

    with pytest.raises(ValueError, match="square"):
        callback.info()


def test_ungrouped_run_still_produces_a_square_matrix(dataset):
    callback = ConceptMetricCallback(
        base_metric=ConstantBaseMetric(),
        summarized_metrics=[FinalStepAverage()],
        stepwise_metrics=[ForgettingMeasure()],
    )

    _run_scenario(dataset, [callback])
    info = callback.info()["concept_metric_callback_Constant"]

    assert info["concepts_order"] == CATEGORIES
    assert info["test_order"] == CATEGORIES
    assert len(info["stepwise_metrics"]["ForgettingMeasure"]) == len(CATEGORIES)
