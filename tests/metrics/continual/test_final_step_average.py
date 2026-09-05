import numpy as np
import pytest

from pyclad.metrics.continual.final_step_average import FinalStepAverage

parameters = [
    # Single concept, single step
    ([[0.5]], 0.5),
    # Square matrix: only the last row counts
    ([[0.9, 0.9], [0.4, 0.6]], 0.5),
    # Rectangular T x N matrix (step schedule "2-1" over 3 categories)
    ([[0.9, 0.8, 0.1], [0.6, 0.4, 0.8]], 0.6),
    # A single training step evaluated on every category
    ([[0.2, 0.4, 0.6]], 0.4),
]


def test_empty_matrix():
    assert FinalStepAverage().compute([]) == 0.0


def test_matrix_without_columns():
    assert FinalStepAverage().compute([[]]) == 0.0


def test_name():
    assert FinalStepAverage().name() == "FinalStepAverage"


@pytest.mark.parametrize("matrix, expected", parameters)
def test_averaging_the_final_row(matrix, expected):
    assert FinalStepAverage().compute(matrix) == pytest.approx(expected)


def test_ignoring_nan_entries():
    assert FinalStepAverage().compute([[0.1, 0.2], [0.4, np.nan]]) == pytest.approx(0.4)
