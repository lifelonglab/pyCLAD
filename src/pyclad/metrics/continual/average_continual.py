import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)


class ContinualAverage(ConceptLevelMetric):

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        concepts_no = len(metric_matrix)
        if concepts_no == 0:
            return 0
        values = []

        for learned_task in range(concepts_no):
            for evaluated_task in range(learned_task + 1):
                values.append(metric_matrix[learned_task][evaluated_task])

        return np.mean(values)

    def name(self) -> str:
        return "ContinualAverage"
