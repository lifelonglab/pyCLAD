import pytest

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix
from pyclad.metrics.continual.final_step_forgetting_measure import (
    FinalStepForgettingMeasure,
)
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure

parameters = [
    # A single concept cannot be forgotten
    ([[0.5]], 0.0),
    # Basic forgetting; the last learned concept is skipped
    ([[0.8, 0.6], [0.2, 0.6]], 0.6),
    # Negative forgetting (the earlier concept improved)
    ([[0.5, 0.5], [0.9, 0.5]], -0.4),
    # The best performance is taken over the steps since the concept was learned, not just the first
    ([[0.7, 0.5, 0.5], [0.9, 0.9, 0.5], [0.6, 0.8, 0.9]], 0.2),
    # A score from before the concept was learned (0.95) is not something to forget
    ([[0.9, 0.95, 0.5], [0.8, 0.9, 0.5], [0.6, 0.8, 0.9]], 0.2),
]


def test_empty_matrix():
    assert FinalStepForgettingMeasure().compute([]) == 0.0


def test_name():
    assert FinalStepForgettingMeasure().name() == "FinalStepForgettingMeasure"


def test_raises_exception_when_matrix_not_square():
    with pytest.raises(ValueError, match="square"):
        FinalStepForgettingMeasure().compute([[0.9, 0.5, 0.5], [0.8, 0.9, 0.5]])


@pytest.mark.parametrize("matrix,expected_result", parameters)
def test_metric_calculation(matrix: ConceptLevelMatrix, expected_result: float):
    assert FinalStepForgettingMeasure().compute(matrix) == pytest.approx(expected_result, rel=1e-9)


def test_agrees_with_last_step_of_forgetting_measure():
    matrix = [[0.9, 0.5, 0.5], [0.8, 0.9, 0.5], [0.6, 0.8, 0.9]]
    assert FinalStepForgettingMeasure().compute(matrix) == pytest.approx(ForgettingMeasure().compute(matrix)[-1])


def test_differs_from_forgetting_measure_when_a_concept_scored_higher_before_it_was_learned():
    matrix = [[0.9, 0.95, 0.5], [0.8, 0.9, 0.5], [0.6, 0.8, 0.9]]
    assert FinalStepForgettingMeasure().compute(matrix) == pytest.approx(0.2)
    assert ForgettingMeasure().compute(matrix)[-1] == pytest.approx(0.225)
