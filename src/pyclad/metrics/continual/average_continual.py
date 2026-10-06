from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    mean_or_nan,
    validate_square_matrix,
)


class ContinualAverage(SummarizedMetric):

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())
        concepts_no = len(metric_matrix)
        values = []

        for learned_task in range(concepts_no):
            for evaluated_task in range(learned_task + 1):
                values.append(metric_matrix[learned_task][evaluated_task])

        return mean_or_nan(values)

    def name(self) -> str:
        return "ContinualAverage"
