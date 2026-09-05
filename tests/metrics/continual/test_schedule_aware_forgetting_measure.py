import numpy as np
import pytest

from pyclad.metrics.continual.schedule_aware_forgetting_measure import (
    ScheduleAwareForgettingMeasure,
)

# Rectangular 2 x 3 matrix produced by the step schedule "2-1" over three categories.
RECTANGULAR = [
    [0.9, 0.8, 0.1],
    [0.6, 0.4, 0.8],
]
RECTANGULAR_FIRST_SEEN = [0, 0, 1]

SQUARE = [
    [0.9, 0.5, 0.1],
    [0.6, 0.8, 0.2],
    [0.3, 0.4, 0.7],
]
SQUARE_FIRST_SEEN = [0, 1, 2]


def test_name():
    assert ScheduleAwareForgettingMeasure().name() == "ScheduleAwareForgettingMeasure"


def test_empty_matrix():
    assert ScheduleAwareForgettingMeasure().compute([], []) == 0.0


def test_single_training_step_has_no_forgetting():
    assert ScheduleAwareForgettingMeasure().compute([[0.5, 0.6]], [0, 0]) == 0.0


def test_forgetting_on_a_rectangular_matrix():
    # col0: max(rows [0, 0]) - final = 0.9 - 0.6 = 0.3
    # col1: max(rows [0, 0]) - final = 0.8 - 0.4 = 0.4
    # col2: first seen at the last step -> skipped
    result = ScheduleAwareForgettingMeasure().compute(RECTANGULAR, RECTANGULAR_FIRST_SEEN)

    assert result == pytest.approx(0.35)


def test_forgetting_on_a_square_matrix():
    # col0: max(0.9, 0.6) - 0.3 = 0.6
    # col1: max(0.8) - 0.4 = 0.4
    # col2: first seen at the last step -> skipped
    result = ScheduleAwareForgettingMeasure().compute(SQUARE, SQUARE_FIRST_SEEN)

    assert result == pytest.approx(0.5)


def test_skipping_columns_first_seen_at_or_after_the_final_step():
    # Every column enters training only at the final step: forgetting is undefined.
    result = ScheduleAwareForgettingMeasure().compute(RECTANGULAR, [1, 1, 1])

    assert result == 0.0


def test_restricting_history_to_rows_after_the_column_was_first_seen():
    matrix = [
        [0.99, 0.99],  # pre-training row for col1, must be ignored
        [0.50, 0.60],
        [0.40, 0.50],
    ]

    result = ScheduleAwareForgettingMeasure().compute(matrix, [0, 1])

    # col0: max(0.99, 0.50) - 0.40 = 0.59 ; col1: max(0.60) - 0.50 = 0.10
    assert result == pytest.approx(0.345)


def test_ignoring_nan_entries():
    matrix = [
        [np.nan, 0.8],
        [0.9, 0.6],
        [0.5, 0.4],
    ]

    result = ScheduleAwareForgettingMeasure().compute(matrix, [0, 0])

    # col0: max(0.9) - 0.5 = 0.4 ; col1: max(0.8, 0.6) - 0.4 = 0.4
    assert result == pytest.approx(0.4)


def test_rejecting_first_seen_steps_of_the_wrong_length():
    with pytest.raises(ValueError, match="first_seen_steps"):
        ScheduleAwareForgettingMeasure().compute(RECTANGULAR, [0, 0])
