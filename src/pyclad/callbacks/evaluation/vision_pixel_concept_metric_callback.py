from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.data.vision_concept import VisionConcept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import SummarizedMetric
from pyclad.output.output_writer import InfoProvider


class VisionPixelConceptMetricCallback(Callback, InfoProvider):
    """Pixel-level evaluation callback.

    Delegates metric computation to internal :class:`ConceptMetricCallback`
    instances.  On each ``after_evaluation`` call it obtains pixel-level
    score maps from the model and reads ground-truth masks from the
    evaluated :class:`VisionConcept`.
    """

    def __init__(
        self,
        strategy: Any,
        base_metrics: Sequence[BaseMetric],
        summarized_metrics: Iterable[SummarizedMetric] = (),
    ):
        self._strategy = strategy
        self._callbacks = [
            ConceptMetricCallback(base_metric=m, summarized_metrics=summarized_metrics) for m in base_metrics
        ]
        self._has_evaluations = False

    def before_scenario(self, *args, **kwargs):
        for cb in self._callbacks:
            cb.before_scenario(*args, **kwargs)

    def after_training(self, learned_concept: Concept, *args, **kwargs):
        for cb in self._callbacks:
            cb.after_training(learned_concept)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        *args,
        **kwargs,
    ):
        model = _resolve_model(self._strategy, evaluated_concept.name)
        score_maps_fn = getattr(model, "score_maps", None)
        if not callable(score_maps_fn):
            return

        score_maps = np.asarray(score_maps_fn(evaluated_concept.data))

        if not (isinstance(evaluated_concept, VisionConcept) and evaluated_concept.masks is not None):
            return

        self._has_evaluations = True
        for cb in self._callbacks:
            cb.after_evaluation(
                evaluated_concept=evaluated_concept,
                y_true=evaluated_concept.masks,
                y_pred=np.asarray([], dtype=np.uint8),
                anomaly_scores=score_maps,
            )

    def after_concept_processing(self, *args, **kwargs):
        for cb in self._callbacks:
            cb.after_concept_processing(*args, **kwargs)

    def after_scenario(self, *args, **kwargs):
        for cb in self._callbacks:
            cb.after_scenario(*args, **kwargs)

    def info(self) -> Dict[str, Any]:
        if not self._has_evaluations:
            return {}
        result: Dict[str, Any] = {}
        for cb in self._callbacks:
            concepts_with_masks = set()
            for evals in cb._metric_matrix.values():
                concepts_with_masks.update(evals.keys())

            original_concepts = cb._learned_concepts
            cb._learned_concepts = [c for c in original_concepts if c in concepts_with_masks]
            try:
                cb_info = cb.info()
            finally:
                cb._learned_concepts = original_concepts

            for key, value in cb_info.items():
                pixel_key = key.replace("concept_metric_callback_", "pixel_concept_metric_callback_")
                value["evaluation_level"] = "pixel"
                result[pixel_key] = value
        return result


def _resolve_model(strategy: Any, concept_name: str):
    models = getattr(strategy, "_models", None)
    if isinstance(models, dict) and concept_name in models:
        return models[concept_name]

    current_model = getattr(strategy, "current_model", None)
    if callable(current_model):
        model = current_model()
        if model is not None:
            return model

    model = getattr(strategy, "_model", None)
    if model is not None:
        return model

    raise AttributeError(f"Could not resolve model from strategy '{type(strategy).__name__}'")
