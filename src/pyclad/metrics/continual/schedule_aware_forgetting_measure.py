from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    is_nan,
    validate_first_seen_steps,
)


class ScheduleAwareForgettingMeasure(ScheduleAwareMetric):
    """Forgetting Measure restricted to the rows in which a category had already been trained.

    For each evaluated category ``k`` first trained at step ``s_k <= T - 2``:

    ``f_k = max_{j in [s_k, T - 2]} M[j][k] - M[T - 1][k]``

    Columns with ``s_k >= T - 1`` are skipped: a category that only enters training at the
    final step cannot have been forgotten yet.

    This restriction is what separates the metric from :class:`ForgettingMeasure`. On a
    ``T x N`` matrix the rows above ``s_k`` describe the model *before* it ever saw the
    category, so ``peak - final`` there measures improvement, not forgetting, and averaging
    it in drags the result toward zero or below. Lower is better.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps: Sequence[int]) -> float:
        rows = len(metric_matrix)
        if rows < 2:
            return 0.0
        validate_first_seen_steps(metric_matrix, first_seen_steps, self.name())

        last_train_row = rows - 1
        values = []
        for column, first_seen in enumerate(first_seen_steps):
            if int(first_seen) >= last_train_row:
                continue

            history = [
                metric_matrix[row][column]
                for row in range(int(first_seen), last_train_row)
                if not is_nan(metric_matrix[row][column])
            ]
            final = metric_matrix[last_train_row][column]
            if not history or is_nan(final):
                continue

            values.append(max(history) - final)

        return float(np.mean(values)) if values else 0.0

    def name(self) -> str:
        return "ScheduleAwareForgettingMeasure"
