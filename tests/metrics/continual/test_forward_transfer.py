import pytest

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix
from pyclad.metrics.continual.forward_transfer import ForwardTransfer

parameters = [
    ([[0.5]], 0),
    ([[0.5, 0.1], [0.2, 0.8]], 0.1),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 11 / 3),
    ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 1.0),
    ([[0.8, 0.0, 0.0], [0.5, 1, 0.0], [0.4, 0.8, 0.9]], 0),
]


def test_empty_matrix():
    metric = ForwardTransfer()
    assert metric.compute([]) == 0


def test_raises_exception_when_matrix_not_square():
    metric = ForwardTransfer()
    with pytest.raises(IndexError):
        metric.compute([[1, 1], [1, 1, 1], [1, 1]])


@pytest.mark.parametrize("matrix,expected_result", parameters)
def test_metric_calculation(matrix: ConceptLevelMatrix, expected_result: float):
    metric = ForwardTransfer()
    assert metric.compute(matrix) == expected_result
