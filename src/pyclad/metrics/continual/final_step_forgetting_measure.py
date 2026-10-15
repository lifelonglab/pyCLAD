from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    validate_square_matrix,
)
from pyclad.metrics.continual.schedule_aware_forgetting_measure import (
    ScheduleAwareForgettingMeasure,
)


class FinalStepForgettingMeasure(SummarizedMetric):
    """Forgetting after the last training step, as a single value.

    For every concept ``j`` except the one learned last, takes the best performance observed on it from the
    step it was learned onwards and subtracts its performance after the final step ``N - 1``:

    ``f_j = max_{i in [j, N - 2]} M[i][j] - M[N - 1][j]``,  ``FM = mean_{j < N - 1} f_j``

    The last learned concept is skipped, since it has had no chance to be forgotten. The result is ``NaN``
    when any value it reads is ``NaN``, or when there is no earlier concept to measure (a single-concept
    scenario). Higher means more forgetting; values below 0 mean earlier concepts improved.

    Differs from :class:`~pyclad.metrics.continual.forgetting_measure.ForgettingMeasure` in two ways: it
    reports one value for the whole scenario instead of one per step, and the best performance is searched
    only from the step at which the concept was learned, so a score the model had on a concept *before*
    training on it never counts as something to forget.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())
        steps = range(len(metric_matrix))
        return ScheduleAwareForgettingMeasure().compute(metric_matrix, list(steps))

    def name(self) -> str:
        return "FinalStepForgettingMeasure"
