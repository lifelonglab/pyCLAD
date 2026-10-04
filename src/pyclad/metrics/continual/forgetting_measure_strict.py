from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    validate_square_matrix,
)
from pyclad.metrics.continual.schedule_aware_forgetting_measure import (
    ScheduleAwareForgettingMeasure,
)


class ForgettingMeasureStrict(SummarizedMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())
        steps = range(len(metric_matrix))
        return ScheduleAwareForgettingMeasure().compute(metric_matrix, list(steps))

    def name(self) -> str:
        return "ForgettingMeasureStrict"
