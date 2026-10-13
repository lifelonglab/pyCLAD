import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    StepwiseConceptMetric,
    validate_square_matrix,
)


class ForgettingMeasure(StepwiseConceptMetric):
    """Forgetting Measure (FM) measures the average performance drop due to forgetting in continual learning.

    After learning concept ``k``, the forgetting of each previously learned concept ``j < k`` is the difference
    between the best performance observed on it at any earlier step and its current performance:

    ``f_{j,k} = max_{i < k} M[i][j] - M[k][j]``,  ``FM_k = mean_{j < k} f_{j,k}``

    The concept just learned is not included: it has had no chance to be forgotten, and its "earlier" performance
    was measured before the model was trained on it. Nothing can be forgotten after the first concept, so the
    first value is always 0.

    FM is between [-1, 1], where a higher value indicates more forgetting, while values below 0 indicate improvement in
    performance on previously learned concepts after learning new ones.

    Reports one value per training step. For a single value describing the end of the scenario, see
    :class:`~pyclad.metrics.continual.final_step_forgetting_measure.FinalStepForgettingMeasure`.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix) -> list[float]:
        """Compute the forgetting measure over the concept-level performance matrix.

        Returns:
            Mean FM after learning each concepts.
        """
        validate_square_matrix(metric_matrix, self.name())

        concepts_no = len(metric_matrix)

        if concepts_no == 0:
            return []

        results: list[float] = []
        for learned_task in range(concepts_no):
            if learned_task == 0:
                results.append(0.0)
                continue
            forgetting_after_learning_task = []
            for evaluated_task in range(learned_task):
                previous_max = max(metric_matrix[t][evaluated_task] for t in range(learned_task))
                current_value = metric_matrix[learned_task][evaluated_task]
                forgetting_after_learning_task.append(previous_max - current_value)
            results.append(float(np.mean(forgetting_after_learning_task)))

        return results

    def name(self) -> str:
        return "ForgettingMeasure"
