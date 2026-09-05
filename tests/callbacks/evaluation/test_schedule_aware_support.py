"""Direct tests for the helper both schedule-aware callbacks share.

Testing it on its own is the point of factoring it out: the image-level and pixel-level
callbacks then exercise one implementation rather than a copy each.
"""

import numpy as np
import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import ScheduleAwareSupport
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.grouping import apply_step_schedule
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
)

CATEGORIES = ["a", "b", "c"]
MATRIX = [[0.9, 0.8, 0.1], [0.6, 0.4, 0.7]]


class RecordingMetric(ScheduleAwareMetric):
    def __init__(self):
        self.received_first_seen = None

    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps) -> float:
        self.received_first_seen = list(first_seen_steps)
        return 0.5

    def name(self) -> str:
        return "Recording"


def _scheduled_dataset(schedule="2-1"):
    dataset = ConceptsDataset(
        name="toy",
        train_concepts=[Concept(name=c, data=np.zeros((2, 1))) for c in CATEGORIES],
        test_concepts=[Concept(name=c, data=np.zeros((2, 1)), labels=np.zeros(2)) for c in CATEGORIES],
    )
    return apply_step_schedule(dataset, schedule)


def test_resolving_the_mapping_from_a_plain_dict():
    support = ScheduleAwareSupport([RecordingMetric()], {"a": 0, "b": 0, "c": 1})

    assert support.first_seen_step() == {"a": 0, "b": 0, "c": 1}


def test_resolving_the_mapping_from_a_step_scheduled_dataset():
    support = ScheduleAwareSupport([RecordingMetric()], _scheduled_dataset())

    assert support.first_seen_step() == {"a": 0, "b": 0, "c": 1}


def test_reading_the_dataset_mapping_at_report_time_rather_than_at_construction():
    """A dataset is read when the payload is built, so the two cannot drift apart."""
    dataset = _scheduled_dataset()
    support = ScheduleAwareSupport([RecordingMetric()], dataset)

    first = support.first_seen_step()
    second = support.first_seen_step()

    assert first == second == dataset.first_seen_step()


def test_aligning_the_steps_to_the_column_order():
    metric = RecordingMetric()
    support = ScheduleAwareSupport([metric], _scheduled_dataset())

    support.payload_additions(MATRIX, ["c", "a", "b"])

    assert metric.received_first_seen == [1, 0, 0]


def test_payload_reports_the_metrics_and_the_resolved_mapping():
    support = ScheduleAwareSupport([RecordingMetric()], {"a": 0, "b": 0, "c": 1})

    payload = support.payload_additions(MATRIX, CATEGORIES)

    assert payload["schedule_aware_metrics"] == {"Recording": 0.5}
    assert payload["first_seen_step"] == {"a": 0, "b": 0, "c": 1}


def test_rejecting_an_empty_metric_list():
    with pytest.raises(ValueError, match="must not be empty"):
        ScheduleAwareSupport([], {"a": 0})


def test_rejecting_a_missing_mapping():
    with pytest.raises(ValueError, match="first_seen_step"):
        ScheduleAwareSupport([RecordingMetric()], None)


def test_rejecting_a_concept_absent_from_the_mapping():
    support = ScheduleAwareSupport([RecordingMetric()], {"a": 0, "b": 0})

    with pytest.raises(ValueError, match="'c'"):
        support.payload_additions(MATRIX, CATEGORIES)
