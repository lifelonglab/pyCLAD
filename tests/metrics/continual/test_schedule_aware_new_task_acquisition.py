import numpy as np
import pytest

from pyclad.metrics.continual.schedule_aware_new_task_acquisition import (
    ScheduleAwareNewTaskAcquisition,
)

RECTANGULAR = [
    [0.9, 0.8, 0.1],
    [0.6, 0.4, 0.8],
]

SQUARE = [
    [0.9, 0.5, 0.1],
    [0.6, 0.8, 0.2],
    [0.3, 0.4, 0.7],
]


def test_name():
    assert ScheduleAwareNewTaskAcquisition().name() == "ScheduleAwareNewTaskAcquisition"


def test_empty_matrix():
    assert ScheduleAwareNewTaskAcquisition().compute([], []) == 0.0


def test_reading_the_value_at_the_step_each_category_was_first_seen():
    # M[0][0]=0.9, M[0][1]=0.8, M[1][2]=0.8
    result = ScheduleAwareNewTaskAcquisition().compute(RECTANGULAR, [0, 0, 1])

    assert result == pytest.approx(0.8333333, abs=1e-6)


def test_identity_first_seen_reads_the_diagonal():
    result = ScheduleAwareNewTaskAcquisition().compute(SQUARE, [0, 1, 2])

    assert result == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_ignoring_nan_entries():
    matrix = [
        [np.nan, 0.8],
        [0.6, 0.4],
    ]

    result = ScheduleAwareNewTaskAcquisition().compute(matrix, [0, 0])

    assert result == pytest.approx(0.8)


def test_rejecting_first_seen_steps_of_the_wrong_length():
    with pytest.raises(ValueError, match="first_seen_steps"):
        ScheduleAwareNewTaskAcquisition().compute(RECTANGULAR, [0, 0])


def test_rejecting_first_seen_step_outside_the_matrix():
    with pytest.raises(ValueError, match="out of range"):
        ScheduleAwareNewTaskAcquisition().compute(RECTANGULAR, [0, 0, 5])
