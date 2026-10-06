from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    mean_or_nan,
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

    The result is ``NaN`` when any value it reads is ``NaN``, or when no category can have been
    forgotten yet.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps: Sequence[int]) -> float:
        rows = len(metric_matrix)
        if rows < 2:
            return mean_or_nan([])
        validate_first_seen_steps(metric_matrix, first_seen_steps, self.name())

        last_train_row = rows - 1
        values = []
        for column, first_seen in enumerate(first_seen_steps):
            if int(first_seen) >= last_train_row:
                continue

            # np.max, unlike the builtin, returns NaN whenever one of the values is NaN.
            history = [metric_matrix[row][column] for row in range(int(first_seen), last_train_row)]
            values.append(np.max(history) - metric_matrix[last_train_row][column])

        return mean_or_nan(values)

    def name(self) -> str:
        return "ScheduleAwareForgettingMeasure"
