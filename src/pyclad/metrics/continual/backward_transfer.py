import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)


class BackwardTransfer(ConceptLevelMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        concepts_no = len(metric_matrix)

        values = []
        for i in range(concepts_no):
            for j in range(i + 1, concepts_no):
                values.append(metric_matrix[j][i] - metric_matrix[j - 1][i])

        return np.mean(values) if len(values) > 0 else 0

    def name(self) -> str:
        return "BackwardTransfer"
