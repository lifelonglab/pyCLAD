from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
)
from pyclad.models.model import Model
from pyclad.output.output_writer import InfoProvider
from pyclad.strategies.baselines.mste import MSTE
from pyclad.strategies.strategy import Strategy
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.models.base import VisionModel


class VisionPixelConceptMetricCallback(Callback, InfoProvider):
    """Pixel-level evaluation callback comparing model score maps against ground-truth masks."""

    def __init__(
        self,
        strategy: Strategy,
        base_metrics: Sequence[BaseMetric],
        summarized_metrics: Iterable[SummarizedMetric] = (),
    ):
        self._strategy = strategy
        self._base_metrics: List[BaseMetric] = list(base_metrics)
        self._summarized_metrics: List[SummarizedMetric] = list(summarized_metrics)
        self._metric_matrices: Dict[str, Dict[str, Dict[str, float]]] = {
            metric.name(): defaultdict(dict) for metric in self._base_metrics
        }
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
        *args,
        **kwargs,
    ) -> None:
        if not (isinstance(evaluated_concept, VisionConcept) and evaluated_concept.masks is not None):
            return

        model = self._resolve_model(evaluated_concept.name)
        if not isinstance(model, VisionModel):
            return

        score_maps = np.asarray(model.score_maps(evaluated_concept.data))
        learned = self._learned_concepts[-1]

        if evaluated_concept.name not in self._evaluated_concepts:
            self._evaluated_concepts.append(evaluated_concept.name)

        for base_metric in self._base_metrics:
            value = base_metric.compute(
                anomaly_scores=score_maps,
                y_true=evaluated_concept.masks,
                y_pred=np.asarray([], dtype=np.uint8),
            )
            self._metric_matrices[base_metric.name()][learned][evaluated_concept.name] = value

    def info(self) -> Dict[str, Any]:
        if not self._evaluated_concepts:
            return {}

        ordered = list(self._evaluated_concepts)
        result: Dict[str, Any] = {}
        for base_metric in self._base_metrics:
            matrix = self._metric_matrices[base_metric.name()]
            dense = self._to_dense_matrix(matrix, ordered)
            result[f"pixel_concept_metric_callback_{base_metric.name()}"] = {
                "base_metric_name": base_metric.name(),
                "metrics": {m.name(): m.compute(dense) for m in self._summarized_metrics},
                "concepts_order": ordered,
                "metric_matrix": matrix,
                "evaluation_level": "pixel",
            }
        return result

    def _resolve_model(self, concept_name: str) -> Optional[Model]:
        if isinstance(self._strategy, MSTE):
            return self._strategy._models.get(concept_name)
        return getattr(self._strategy, "_model", None)

    @staticmethod
    def _to_dense_matrix(
        metric_matrix: Dict[str, Dict[str, float]],
        concepts_order: List[str],
    ) -> ConceptLevelMatrix:
        if not concepts_order:
            return [[]]
        return [[metric_matrix[learned][evaluated] for evaluated in concepts_order] for learned in concepts_order]
