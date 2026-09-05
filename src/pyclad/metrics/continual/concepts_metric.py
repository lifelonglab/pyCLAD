import abc
from typing import List, Sequence, Tuple

import numpy as np

ConceptLevelMatrix = List[List[float]]  # ConceptLevelMatrix[learned_concept][evaluated_concept]


class SummarizedMetric(abc.ABC):
    """Base class for metrics that transform the concept-level metric matrix to summarized single value metric."""

    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...


class StepwiseConceptMetric(abc.ABC):
    """Base class for metrics that transform the concept-level metric matrix a separate value for each learned concept."""

    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix) -> List[float]: ...

    @abc.abstractmethod
    def name(self) -> str: ...


class ScheduleAwareMetric(abc.ABC):
    """Base class for metrics computed over a rectangular (``T x N``) concept-level matrix.

    Such a matrix appears when training steps group several categories (see
    :mod:`pyclad.data.grouping`) while evaluation stays per category. Rows are training
    steps, columns are evaluated categories, and the two axes no longer share names.

    Interpreting such a matrix requires knowing, for each column, the training step at
    which that category was first trained on -- rows above it describe the model *before*
    it ever saw the category, which is a different quantity from forgetting. That mapping
    is supplied as ``first_seen_steps``, aligned with the matrix columns; build it with
    :func:`pyclad.data.grouping.compute_first_seen_step`.
    """

    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix, first_seen_steps: Sequence[int]) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...


def matrix_shape(metric_matrix: ConceptLevelMatrix) -> Tuple[int, int]:
    """Return ``(rows, columns)`` of a concept-level matrix."""
    rows = len(metric_matrix)
    return rows, len(metric_matrix[0]) if rows > 0 else 0


def validate_square_matrix(metric_matrix: ConceptLevelMatrix, metric_name: str) -> None:
    """Reject a rectangular concept-level matrix.

    Metrics that walk the diagonal or the triangles of the matrix are only meaningful when
    every training step corresponds to exactly one evaluated concept. On a ``T x N`` matrix
    produced by a step schedule they would silently return a number computed over cells
    that do not mean what the formula assumes.

    :raises ValueError: when the matrix has rows of unequal length, or is non-empty and not
        square. An empty matrix (``[]``) is accepted; ``[[]]`` is not, since one row with no
        columns is not square.
    """
    row_lengths = {len(row) for row in metric_matrix}
    if len(row_lengths) > 1:
        raise ValueError(
            f"{metric_name} requires a square concept-level matrix, but its rows have "
            f"unequal lengths: {sorted(row_lengths)}."
        )

    rows, columns = matrix_shape(metric_matrix)
    if rows > 0 and rows != columns:
        raise ValueError(
            f"{metric_name} requires a square concept-level matrix, got {rows}x{columns}. "
            "A rectangular matrix comes from step-scheduled training, where rows are training "
            "steps and columns are categories; use the schedule-aware metrics instead."
        )


def validate_first_seen_steps(
    metric_matrix: ConceptLevelMatrix, first_seen_steps: Sequence[int], metric_name: str
) -> None:
    """Check that ``first_seen_steps`` aligns with the columns of ``metric_matrix``.

    :raises ValueError: when the lengths differ, or an index falls outside the matrix rows.
    """
    rows, columns = matrix_shape(metric_matrix)
    if len(first_seen_steps) != columns:
        raise ValueError(
            f"{metric_name}: first_seen_steps has length {len(first_seen_steps)} "
            f"but the matrix has {columns} columns."
        )
    for column, first_seen in enumerate(first_seen_steps):
        if not 0 <= int(first_seen) < rows:
            raise ValueError(
                f"{metric_name}: first_seen_steps[{column}] = {first_seen} is out of range "
                f"for a matrix with {rows} training steps."
            )


def is_nan(value: float) -> bool:
    """True when ``value`` is NaN. Metric matrices carry NaN where a base metric was undefined."""
    return bool(np.isnan(value))
