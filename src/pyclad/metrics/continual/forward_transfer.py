from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    mean_or_nan,
    validate_square_matrix,
)


class ForwardTransfer(SummarizedMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())
        concepts_no = len(metric_matrix)

        values = []
        for i in range(concepts_no):
            for j in range(i + 1, concepts_no):
                values.append(metric_matrix[i][j])

        return mean_or_nan(values)

    def name(self) -> str:
        return "ForwardTransfer"
