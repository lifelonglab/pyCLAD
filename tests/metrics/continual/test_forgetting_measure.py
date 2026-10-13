import pytest

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure

parameters = [
    # Simple one concept
    ([[0.5]], [0.0]),
    # Basic forgetting
    ([[0.8, 0.6], [0.2, 0.6]], [0.0, 0.6]),
    # Negative forgetting (improvement in performance on previously learned concept after learning new one)
    ([[0.5, 0.5], [0.9, 0.5]], [0.0, -0.4]),
    # The just-learned concept is excluded, even when it scores below its pre-training value
    ([[0.8, 0.6], [0.4, 0.3]], [0.0, 0.4]),
    # Forgetting on one concept but improvement on the other
    ([[0.9, 0.5, 0.5], [0.9, 0.5, 0.5], [0.5, 0.9, 0.5]], [0.0, 0.0, 0.0]),
    # no forgetting — constant performance across all steps
    ([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [0.0, 0.0, 0.0]),
    # forgetting accumulates as more concepts are learned
    ([[0.9, 0.9, 0.9], [0.6, 0.9, 0.9], [0.3, 0.6, 0.9]], [0.0, 0.3, 0.45]),
    # the gain on the just-learned concept does not offset forgetting of the earlier ones
    ([[0.9, 0.5, 0.5], [0.8, 0.9, 0.5], [0.6, 0.8, 0.9]], [0.0, 0.1, 0.2]),
]


def test_empty_matrix():
    metric = ForgettingMeasure()
    assert metric.compute([]) == []


def test_name():
    metric = ForgettingMeasure()
    assert metric.name() == "ForgettingMeasure"


@pytest.mark.parametrize("matrix,expected_result", parameters)
def test_metric_calculation(matrix: ConceptLevelMatrix, expected_result: float):
    metric = ForgettingMeasure()
    assert metric.compute(matrix) == pytest.approx(expected_result, rel=1e-9)
