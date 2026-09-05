import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    StepwiseConceptMetric,
    validate_square_matrix,
)


class ForgettingMeasure(StepwiseConceptMetric):
    """Forgetting Measure (FM) measures the average performance drop due to forgetting in continual learning.

    For each (learned_concept, evaluated_concept) pair, computes the difference between the best previously
    observed performance on the evaluated concept and the current performance after learning the latest concept.

    FM is between [-1, 1], where a higher value indicates more forgetting, while values below 0 indicate improvement in
    performance on previously learned concepts after learning new ones.
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
            for evaluated_task in range(learned_task + 1):
                previous_max = max(metric_matrix[t][evaluated_task] for t in range(learned_task))
                current_value = metric_matrix[learned_task][evaluated_task]
                forgetting_after_learning_task.append(previous_max - current_value)
            results.append(float(np.mean(forgetting_after_learning_task)))

        return results

    def name(self) -> str:
        return "ForgettingMeasure"
