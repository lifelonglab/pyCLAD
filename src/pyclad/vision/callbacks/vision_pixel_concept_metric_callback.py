from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.concept_metric_evaluation import (
    FirstSeenStepSource,
    ScheduleAwareSupport,
    build_dense_matrix,
    resolve_column_order,
)
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    SummarizedMetric,
)
from pyclad.output.output_writer import InfoProvider
from pyclad.vision.data.vision_concept import VisionConcept


class VisionPixelConceptMetricCallback(Callback, InfoProvider):
    """Pixel-level variant of :class:`pyclad.callbacks.evaluation.concept_metric_evaluation.ConceptMetricCallback`.

    Same shape as ``ConceptMetricCallback`` — one ``base_metric`` per callback
    instance, optional ``summarized_metrics`` — but reads per-pixel ``score_maps``
    (forwarded by the scenario as the ``score_maps`` kwarg on
    ``after_evaluation`` whenever the strategy's ``predict()`` returns
    :class:`pyclad.vision.prediction_results.VisionPredictionResults`) and
    ground-truth masks from the evaluated :class:`VisionConcept`.

    Skips silently when ``score_maps`` is absent (i.e. the strategy returned a
    plain :class:`pyclad.output.prediction_results.PredictionResults`), when the
    evaluated concept is not a :class:`VisionConcept`, or when the concept
    carries no masks. Note that skipping *some* pairs leaves the matrix incomplete,
    which :meth:`info` reports as an error rather than papering over.

    For step-scheduled training (see :mod:`pyclad.data.grouping`), where rows are grouped
    training steps and columns stay per category, use
    :class:`ScheduleAwareVisionPixelConceptMetricCallback`.
    """

    def __init__(
        self,
        base_metric: BaseMetric,
        summarized_metrics: Iterable[SummarizedMetric] = (),
    ):
        self._base_metric = base_metric
        self._summarized_metrics: List[SummarizedMetric] = list(summarized_metrics)
        self._metric_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._learned_concepts: List[str] = []
        self._evaluated_concepts: List[str] = []

    def after_training(self, learned_concept: Concept, *args, **kwargs) -> None:
        self._learned_concepts.append(learned_concept.name)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        score_maps: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> None:
        if score_maps is None or not isinstance(evaluated_concept, VisionConcept) or evaluated_concept.masks is None:
            return

        learned = self._learned_concepts[-1]
        if evaluated_concept.name not in self._evaluated_concepts:
            self._evaluated_concepts.append(evaluated_concept.name)

        value = self._base_metric.compute(
            anomaly_scores=score_maps,
            y_true=evaluated_concept.masks,
            y_pred=np.asarray([], dtype=np.uint8),
        )
        self._metric_matrix[learned][evaluated_concept.name] = value

    def column_order(self) -> List[str]:
        """Concept order of the matrix columns, as actually used to build it."""
        return resolve_column_order(self._learned_concepts, self._evaluated_concepts)

    def dense_matrix(self) -> ConceptLevelMatrix:
        return build_dense_matrix(self._metric_matrix, self._learned_concepts, self._evaluated_concepts)

    def info(self) -> Dict[str, Any]:
        if not self._evaluated_concepts:
            return {}

        dense = self.dense_matrix()
        return {
            f"pixel_concept_metric_callback_{self._base_metric.name()}": {
                "base_metric_name": self._base_metric.name(),
                "metrics": {m.name(): m.compute(dense) for m in self._summarized_metrics},
                "concepts_order": self._learned_concepts,
                "test_order": self.column_order(),
                "metric_matrix": self._metric_matrix,
                "evaluation_level": "pixel",
            }
        }


class ScheduleAwareVisionPixelConceptMetricCallback(VisionPixelConceptMetricCallback):
    """Pixel-level counterpart of
    :class:`~pyclad.callbacks.evaluation.concept_metric_evaluation.ScheduleAwareConceptMetricCallback`.

    Shares its schedule-aware reporting through
    :class:`~pyclad.callbacks.evaluation.concept_metric_evaluation.ScheduleAwareSupport`, so the
    two levels cannot drift apart.
    """

    def __init__(
        self,
        base_metric: BaseMetric,
        summarized_metrics: Iterable[SummarizedMetric] = (),
        schedule_aware_metrics: Iterable[ScheduleAwareMetric] = (),
        first_seen_step: Optional[FirstSeenStepSource] = None,
    ):
        super().__init__(base_metric, summarized_metrics)
        self._schedule_aware = ScheduleAwareSupport(schedule_aware_metrics, first_seen_step)

    def info(self) -> Dict[str, Any]:
        payload = super().info()
        if not payload:
            return payload
        next(iter(payload.values())).update(
            self._schedule_aware.payload_additions(self.dense_matrix(), self.column_order())
        )
        return payload
