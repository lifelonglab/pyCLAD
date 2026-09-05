from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.data.grouping import StepScheduledConceptsDataset
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    StepwiseConceptMetric,
    SummarizedMetric,
)
from pyclad.output.output_writer import InfoProvider

FirstSeenStepSource = Union[Mapping[str, int], StepScheduledConceptsDataset]


class ConceptMetricCallback(Callback, InfoProvider):
    """Collects a base metric for every (training step, evaluated concept) pair.

    Rows follow the order in which concepts were learned, columns the order in which they were
    evaluated. With one concept per training step both axes hold the same names, columns are
    aligned to the training order, and the matrix is square.

    When training steps group several concepts (see :mod:`pyclad.data.grouping`) the axes stop
    sharing names: rows are the grouped steps while columns stay per concept, so the matrix is
    rectangular. Metrics that need to know when each concept entered training live in
    :class:`ScheduleAwareConceptMetricCallback`.
    """

    def __init__(
        self,
        base_metric: BaseMetric,
        summarized_metrics: Iterable[SummarizedMetric],
        stepwise_metrics: Iterable[StepwiseConceptMetric] = None,
    ):
        self._base_metric: BaseMetric = base_metric
        self._metric_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._learned_concepts: List[str] = []
        self._evaluated_concepts: List[str] = []
        self._summarized_metrics = summarized_metrics
        self._stepwise_metrics = stepwise_metrics if stepwise_metrics is not None else []

    def after_training(self, learned_concept: Concept):
        self._learned_concepts.append(learned_concept.name)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        *args,
        **kwargs,
    ):
        assert (
            evaluated_concept.name not in self._metric_matrix[self._learned_concepts[-1]]
        ), "The same concept should not be evaluated twice after the same learned concept"

        metric_value = self._base_metric.compute(anomaly_scores=anomaly_scores, y_true=y_true, y_pred=y_pred)
        self._metric_matrix[self._learned_concepts[-1]][evaluated_concept.name] = metric_value

        if evaluated_concept.name not in self._evaluated_concepts:
            self._evaluated_concepts.append(evaluated_concept.name)

    def column_order(self) -> List[str]:
        """Concept order of the matrix columns, as actually used to build it."""
        return resolve_column_order(self._learned_concepts, self._evaluated_concepts)

    def dense_matrix(self) -> ConceptLevelMatrix:
        return build_dense_matrix(self._metric_matrix, self._learned_concepts, self._evaluated_concepts)

    def info(self) -> Dict[str, Any]:
        concept_level_matrix = self.dense_matrix()
        return {
            f"concept_metric_callback_{self._base_metric.name()}": {
                "base_metric_name": self._base_metric.name(),
                "metrics": {m.name(): m.compute(concept_level_matrix) for m in self._summarized_metrics},
                "stepwise_metrics": {m.name(): m.compute(concept_level_matrix) for m in self._stepwise_metrics},
                "concepts_order": self._learned_concepts,
                "test_order": self.column_order(),
                "metric_matrix": self._metric_matrix,
            }
        }


class ScheduleAwareConceptMetricCallback(ConceptMetricCallback):
    """:class:`ConceptMetricCallback` that also reports metrics over a step-scheduled matrix.

    Changes nothing about how the matrix is collected, only what is reported from it. Use it when
    training steps group several concepts, so the matrix is rectangular and the square-only
    metrics no longer apply.

    :param schedule_aware_metrics: metrics computed as ``compute(matrix, first_seen_steps)``.
    :param first_seen_step: either a :class:`~pyclad.data.grouping.StepScheduledConceptsDataset`,
        whose mapping is then read at reporting time rather than copied at construction, or a
        plain ``{concept: step}`` mapping for a stream not built by
        :func:`~pyclad.data.grouping.apply_step_schedule`.
    """

    def __init__(
        self,
        base_metric: BaseMetric,
        summarized_metrics: Iterable[SummarizedMetric],
        stepwise_metrics: Iterable[StepwiseConceptMetric] = None,
        schedule_aware_metrics: Iterable[ScheduleAwareMetric] = (),
        first_seen_step: Optional[FirstSeenStepSource] = None,
    ):
        super().__init__(base_metric, summarized_metrics, stepwise_metrics)
        self._schedule_aware = ScheduleAwareSupport(schedule_aware_metrics, first_seen_step)

    def info(self) -> Dict[str, Any]:
        payload = super().info()
        if not payload:
            return payload
        next(iter(payload.values())).update(
            self._schedule_aware.payload_additions(self.dense_matrix(), self.column_order())
        )
        return payload


class ScheduleAwareSupport:
    """The extra reporting that schedule-aware callbacks layer onto a base callback.

    Kept out of the callbacks themselves so the image-level and pixel-level variants -- and any
    base callback added later -- share one implementation instead of a copy each.
    """

    def __init__(
        self,
        metrics: Iterable[ScheduleAwareMetric],
        first_seen_step: Optional[FirstSeenStepSource],
    ):
        self._metrics: List[ScheduleAwareMetric] = list(metrics)
        if not self._metrics:
            raise ValueError("schedule_aware_metrics must not be empty; use the plain callback when there are none.")
        if first_seen_step is None:
            raise ValueError(
                "schedule_aware_metrics require first_seen_step, mapping every evaluated concept to "
                "the training step that first includes it. Pass the StepScheduledConceptsDataset "
                "returned by pyclad.data.grouping.apply_step_schedule, or an explicit mapping."
            )
        self._first_seen_step = first_seen_step

    def first_seen_step(self) -> Dict[str, int]:
        """Resolve the mapping, reading it off the dataset when one was given."""
        if isinstance(self._first_seen_step, StepScheduledConceptsDataset):
            return self._first_seen_step.first_seen_step()
        return dict(self._first_seen_step)

    def payload_additions(self, metric_matrix: ConceptLevelMatrix, evaluated_concepts: Sequence[str]) -> Dict[str, Any]:
        first_seen_steps = align_first_seen_steps(self.first_seen_step(), evaluated_concepts)
        return {
            "first_seen_step": dict(zip(evaluated_concepts, first_seen_steps)),
            "schedule_aware_metrics": {
                metric.name(): metric.compute(metric_matrix, first_seen_steps) for metric in self._metrics
            },
        }


def resolve_column_order(train_order: Sequence[str], test_order: Sequence[str]) -> List[str]:
    """Return the concept order the matrix columns should follow.

    When both axes hold the same concepts the matrix is square and its cells are read by position:
    ``[i][i]`` must mean "concept *i* evaluated after training on concept *i*". The two orders are
    tracked independently, so columns are aligned to the training order to keep that true even
    when a dataset evaluates its concepts in a different order than it trains them.

    When the axes differ -- grouped training steps against per-concept evaluation -- there is no
    positional correspondence to preserve, and the evaluation order is kept.
    """
    if set(train_order) == set(test_order):
        return list(train_order)
    return list(test_order)


def build_dense_matrix(
    metric_matrix: Mapping[str, Mapping[str, float]],
    train_order: Sequence[str],
    test_order: Sequence[str],
) -> ConceptLevelMatrix:
    """Turn the sparse ``{learned: {evaluated: value}}`` mapping into a dense matrix.

    Rows follow ``train_order``; columns follow :func:`resolve_column_order`. The scenario
    evaluates every test concept after every training step, so a missing cell means a broken run
    rather than a legitimately unknown value.

    :raises ValueError: when any (training step, evaluated concept) pair is missing.
    """
    if len(train_order) == 0 or len(test_order) == 0:
        return [[]]

    column_order = resolve_column_order(train_order, test_order)
    values: ConceptLevelMatrix = []
    for learned_concept in train_order:
        row = []
        for evaluated_concept in column_order:
            if evaluated_concept not in metric_matrix.get(learned_concept, {}):
                raise ValueError(
                    f"Concept {evaluated_concept!r} was not evaluated after training step "
                    f"{learned_concept!r}; the concept-level matrix is incomplete."
                )
            row.append(metric_matrix[learned_concept][evaluated_concept])
        values.append(row)
    return values


def align_first_seen_steps(first_seen_step: Mapping[str, int], test_order: Sequence[str]) -> List[int]:
    """Order a ``{concept: first seen step}`` mapping to match the matrix columns.

    :raises ValueError: when an evaluated concept is absent from the mapping.
    """
    missing = [name for name in test_order if name not in first_seen_step]
    if missing:
        raise ValueError(
            f"first_seen_step is missing an entry for evaluated concepts {missing}. It must cover "
            "every evaluated concept. Note that schedule-aware metrics describe the rectangular "
            "matrix produced with group_test=False; with group_test=True both axes hold grouped "
            "step names and these metrics do not apply."
        )
    return [int(first_seen_step[name]) for name in test_order]
