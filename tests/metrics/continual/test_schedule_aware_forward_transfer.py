import numpy as np
import pytest

from pyclad.metrics.continual.schedule_aware_forward_transfer import (
    ScheduleAwareForwardTransfer,
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
    assert ScheduleAwareForwardTransfer().name() == "ScheduleAwareForwardTransfer"


def test_empty_matrix():
    assert np.isnan(ScheduleAwareForwardTransfer().compute([], []))


def test_forward_transfer_on_a_rectangular_matrix():
    # Only col2 has pre-training rows: mean(0.1) = 0.1
    result = ScheduleAwareForwardTransfer().compute(RECTANGULAR, [0, 0, 1])

    assert result == pytest.approx(0.1)


def test_forward_transfer_on_a_square_matrix():
    # col1: mean(0.5) = 0.5 ; col2: mean(0.1, 0.2) = 0.15 ; col0 has no pre-training rows
    result = ScheduleAwareForwardTransfer().compute(SQUARE, [0, 1, 2])

    assert result == pytest.approx(0.325)


def test_undefined_when_every_column_is_trained_from_the_very_first_step():
    result = ScheduleAwareForwardTransfer().compute(RECTANGULAR, [0, 0, 0])

    assert np.isnan(result)


def test_nan_in_a_pre_training_row_makes_the_result_nan():
    matrix = [
        [0.2, np.nan],
        [0.4, 0.6],
        [0.5, 0.5],
    ]

    assert np.isnan(ScheduleAwareForwardTransfer().compute(matrix, [0, 2]))


def test_nan_after_a_category_was_first_seen_is_irrelevant():
    matrix = [
        [0.2, 0.6],
        [0.4, np.nan],
        [0.5, 0.5],
    ]

    # col1: the only pre-training row is the first one
    assert ScheduleAwareForwardTransfer().compute(matrix, [0, 1]) == pytest.approx(0.6)


def test_rejecting_first_seen_steps_of_the_wrong_length():
    with pytest.raises(ValueError, match="first_seen_steps"):
        ScheduleAwareForwardTransfer().compute(RECTANGULAR, [0, 1])
