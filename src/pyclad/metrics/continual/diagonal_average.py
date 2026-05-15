from typing import List

import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    StepwiseConceptMetric,
)


class DiagonalAverage(StepwiseConceptMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> List[float]:
        concepts_no = len(metric_matrix)
        if concepts_no == 0:
            return []

        diag = [metric_matrix[i][i] for i in range(concepts_no)]
        return [float(np.nanmean(diag[: k + 1])) for k in range(concepts_no)]

    def name(self) -> str:
        return "DiagonalAverage"
