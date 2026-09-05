from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    is_nan,
    validate_first_seen_steps,
)


class ScheduleAwareNewTaskAcquisition(ScheduleAwareMetric):
    """Average performance on each category right after it first entered training.

    For each evaluated category ``k`` first trained at step ``s_k``: ``nta_k = M[s_k][k]``.

    This is how well the model acquires a new category before any later step has had a
    chance to interfere, which disentangles two failure modes that a single number hides:
    low acquisition means the category was never learned (plasticity), while high
    acquisition together with high forgetting means it was learned and then lost
    (stability). On a square matrix with the identity schedule this reads the diagonal.

    ``NaN`` entries are ignored. Higher is better.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps: Sequence[int]) -> float:
        if len(metric_matrix) == 0:
            return 0.0
        validate_first_seen_steps(metric_matrix, first_seen_steps, self.name())

        values = [
            float(metric_matrix[int(first_seen)][column])
            for column, first_seen in enumerate(first_seen_steps)
            if not is_nan(metric_matrix[int(first_seen)][column])
        ]

        return float(np.mean(values)) if values else 0.0

    def name(self) -> str:
        return "ScheduleAwareNewTaskAcquisition"
