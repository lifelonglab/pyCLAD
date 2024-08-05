import pytest

from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix

parameters = [
    ([[0.5]], 0.5),
    ([[0.5, 0.1], [0.2, 0.8]], 0.5),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 34 / 6),
    ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 1.0),
    ([[0.8, 0.3, 0.0], [0.5, 1, 0.5], [0.4, 0.8, 0.9]], 4.4 / 6),
]


def test_empty_matrix():
    metric = ContinualAverage()
    assert metric.compute([]) == 0


def test_raises_exception_when_matrix_not_square():
    metric = ContinualAverage()
    with pytest.raises(IndexError):
        metric.compute([[1, 1, 1], [1, 1, 1], [1, 1]])


@pytest.mark.parametrize("matrix,expected_result", parameters)
def test_metric_calculation(matrix: ConceptLevelMatrix, expected_result: float):
    metric = ContinualAverage()
    assert metric.compute(matrix) == expected_result


def test_metric_does_not_depend_on_upper_diagonal():
    matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix2 = [[1, 20, 30], [4, 5, 60], [7, 8, 9]]
    metric = ContinualAverage()
    assert metric.compute(matrix1) == metric.compute(matrix2)
