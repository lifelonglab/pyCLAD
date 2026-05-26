from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
)
from pyclad.output.output_writer import InfoProvider
from pyclad.vision.data.vision_concept import VisionConcept


class VisionPixelConceptMetricCallback(Callback, InfoProvider):
    """Pixel-level variant of :class:`pyclad.callbacks.evaluation.concept_metric_evaluation.ConceptMetricCallback`.

    Same shape as ``ConceptMetricCallback`` — one ``base_metric`` per callback
    instance, optional ``summarized_metrics`` — but reads per-pixel ``score_maps``
    (supplied by :class:`pyclad.vision.scenarios.concept_incremental.VisionConceptIncrementalScenario`
    via the ``score_maps`` kwarg on ``after_evaluation``) and ground-truth masks
    from the evaluated :class:`VisionConcept`.

    Skips silently when the scenario does not provide ``score_maps`` (i.e. when
    running under a non-vision scenario), when the evaluated concept is not a
    :class:`VisionConcept`, or when the concept carries no masks.
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

    def info(self) -> Dict[str, Any]:
        if not self._evaluated_concepts:
            return {}

        ordered = list(self._evaluated_concepts)
        dense = self._to_dense_matrix(self._metric_matrix, ordered)
        return {
            f"pixel_concept_metric_callback_{self._base_metric.name()}": {
                "base_metric_name": self._base_metric.name(),
                "metrics": {m.name(): m.compute(dense) for m in self._summarized_metrics},
                "concepts_order": ordered,
                "metric_matrix": self._metric_matrix,
                "evaluation_level": "pixel",
            }
        }

    @staticmethod
    def _to_dense_matrix(
        metric_matrix: Dict[str, Dict[str, float]],
        concepts_order: List[str],
    ) -> ConceptLevelMatrix:
        if not concepts_order:
            return [[]]
        return [[metric_matrix[learned][evaluated] for evaluated in concepts_order] for learned in concepts_order]
