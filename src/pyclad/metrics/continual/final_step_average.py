import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    is_nan,
)


class FinalStepAverage(SummarizedMetric):
    """Average of the base metric over all evaluated concepts after the last training step.

    This is the ``A-AUROC`` figure reported by CDAD-style continual anomaly detection
    papers: train through the whole sequence, then average across every test concept.

    Works on both square (``N x N``) and rectangular (``T x N``) matrices, since it only
    reads the last row. ``NaN`` entries are ignored. Higher is better.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        if len(metric_matrix) == 0:
            return 0.0

        final_row = [value for value in metric_matrix[-1] if not is_nan(value)]
        return float(np.mean(final_row)) if final_row else 0.0

    def name(self) -> str:
        return "FinalStepAverage"
