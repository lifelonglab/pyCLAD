import numpy as np


from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix, ConceptLevelMatrixMetric


class ForwardTransfer(ConceptLevelMatrixMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        concepts_no = len(metric_matrix)

        values = []
        for i in range(concepts_no):
            for j in range(i+1, concepts_no):
                values.append(metric_matrix[i][j])

        return np.mean(values)

    def name(self) -> str:
        return 'ForwardTransfer'
