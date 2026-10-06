from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    mean_or_nan,
)


class FinalStepAverage(SummarizedMetric):
    """Average of the base metric over all evaluated concepts after the last training step.

    This is the ``A-AUROC`` figure reported by CDAD-style continual anomaly detection
    papers: train through the whole sequence, then average across every test concept.

    Works on both square (``N x N``) and rectangular (``T x N``) matrices, since it only
    reads the last row. The result is ``NaN`` when any entry of that row is ``NaN``. Higher is better.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        if len(metric_matrix) == 0:
            return mean_or_nan([])

        return mean_or_nan(metric_matrix[-1])

    def name(self) -> str:
        return "FinalStepAverage"
