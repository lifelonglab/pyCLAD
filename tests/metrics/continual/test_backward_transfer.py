import math

import pytest

from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix

parameters = [
    ([[0.5]], 0),
    ([[0.5, 0.1], [0.2, 0.8]], -0.3),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3),
    ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 0),
    ([[0.8, 0.0, 0.0], [0.5, 1, 0.0], [0.4, 0.8, 0.9]], -0.2),
]


def test_empty_matrix():
    metric = BackwardTransfer()
    assert metric.compute([]) == 0


def test_raises_exception_when_matrix_not_square():
    metric = BackwardTransfer()
    with pytest.raises(IndexError):
        metric.compute([[1, 1, 1], [1], [1, 1, 1]])


@pytest.mark.parametrize("matrix,expected_result", parameters)
def test_metric_calculation(matrix: ConceptLevelMatrix, expected_result: float):
    metric = BackwardTransfer()
    assert math.isclose(metric.compute(matrix), expected_result, rel_tol=1e-9)
