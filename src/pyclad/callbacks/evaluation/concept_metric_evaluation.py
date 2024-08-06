from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)
from pyclad.output.output_writer import InfoProvider


class ConceptMetricCallback(Callback, InfoProvider):
    def __init__(self, base_metric: BaseMetric, metrics: Iterable[ConceptLevelMetric]):
        self._base_metric: BaseMetric = base_metric
        self._metric_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._learned_concepts: List[str] = []
        self._metrics = metrics

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

    def info(self) -> Dict[str, Any]:

        concept_level_matrix = self._transform_to_ordered_matrix(self._metric_matrix, self._learned_concepts)
        lifelong_learning_metrics = {m.name(): m.compute(concept_level_matrix) for m in self._metrics}

        return {
            f"concept_metric_callback_{self._base_metric.name()}": {
                "base_metric_name": self._base_metric.name(),
                "metrics": lifelong_learning_metrics,
                "concepts_order": self._learned_concepts,
                "metric_matrix": self._metric_matrix,
            }
        }

    @staticmethod
    def _transform_to_ordered_matrix(
        metric_matrix: Dict[str, Dict[str, float]], concepts_order: List[str]
    ) -> ConceptLevelMatrix:
        if len(concepts_order) == 0:
            return [[]]
        values = []

        for learned_concept in concepts_order:
            values.append([])
            for evaluated_concept in concepts_order:
                values[-1].append(metric_matrix[learned_concept][evaluated_concept])

        return values
