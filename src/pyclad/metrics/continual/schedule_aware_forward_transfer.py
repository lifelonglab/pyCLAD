from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    is_nan,
    validate_first_seen_steps,
)


class ScheduleAwareForwardTransfer(ScheduleAwareMetric):
    """Average performance on categories the model has not been trained on yet.

    For each evaluated category ``k`` first trained at step ``s_k > 0``:

    ``fwt_k = mean_{j in [0, s_k - 1]} M[j][k]``

    Columns with ``s_k = 0`` are skipped -- they have no pre-training rows. ``NaN`` entries
    are ignored. Higher is better: it measures how well knowledge from earlier steps
    transfers to categories that are still unseen.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps: Sequence[int]) -> float:
        if len(metric_matrix) == 0:
            return 0.0
        validate_first_seen_steps(metric_matrix, first_seen_steps, self.name())

        per_column_means = []
        for column, first_seen in enumerate(first_seen_steps):
            pre_training_values = [
                metric_matrix[row][column] for row in range(int(first_seen)) if not is_nan(metric_matrix[row][column])
            ]
            if pre_training_values:
                per_column_means.append(float(np.mean(pre_training_values)))

        return float(np.mean(per_column_means)) if per_column_means else 0.0

    def name(self) -> str:
        return "ScheduleAwareForwardTransfer"
