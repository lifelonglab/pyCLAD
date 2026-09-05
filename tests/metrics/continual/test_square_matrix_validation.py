"""Metrics that walk the diagonal or the triangles of the matrix require a square one.

With a step schedule the metric matrix becomes rectangular (T x N, T < N). Without these
guards the classic metrics silently return a number computed over meaningless cells --
the kind of defect that survives review and ends up in a results table.
"""

import pytest

from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure
from pyclad.metrics.continual.forward_transfer import ForwardTransfer

RECTANGULAR = [
    [0.9, 0.8, 0.1],
    [0.6, 0.4, 0.8],
]

SQUARE = [
    [0.9, 0.5],
    [0.6, 0.8],
]

METRICS = [ContinualAverage(), BackwardTransfer(), ForwardTransfer(), ForgettingMeasure()]


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name())
def test_rejecting_rectangular_matrix(metric):
    with pytest.raises(ValueError, match="square"):
        metric.compute(RECTANGULAR)


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name())
def test_accepting_square_matrix(metric):
    metric.compute(SQUARE)


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name())
def test_accepting_empty_matrix(metric):
    metric.compute([])


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name())
def test_rejecting_a_row_without_columns(metric):
    """`[[]]` is not square either: one row, no columns."""
    with pytest.raises(ValueError, match="square"):
        metric.compute([[]])
