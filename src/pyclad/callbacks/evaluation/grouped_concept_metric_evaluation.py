from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
)
from pyclad.output.output_writer import InfoProvider


class GroupedConceptMetricCallback(Callback, InfoProvider):
    def __init__(
        self,
        base_metric: BaseMetric,
        group_by_concept: Mapping[str, str],
        summarized_metrics: Iterable[SummarizedMetric] = (),
    ):
        self._base_metric = base_metric
        self._group_by_concept = dict(group_by_concept)
        self._summarized_metrics: List[SummarizedMetric] = list(summarized_metrics)
        self._learned_groups: List[str] = []
        self._evaluated_groups: List[str] = []
        self._values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def after_training(self, learned_concept: Concept, *args, **kwargs) -> None:
        self._learned_groups.append(learned_concept.name)

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
        value = self._concept_value(evaluated_concept, y_true, y_pred, anomaly_scores, score_maps)
        if value is None:
            return

        group = self._group_by_concept[evaluated_concept.name]
        if group not in self._evaluated_groups:
            self._evaluated_groups.append(group)
        self._values[self._learned_groups[-1]][group].append(value)

    def info(self) -> Dict[str, Any]:
        if not self._evaluated_groups:
            return {}

        group_matrix = {
            learned: {group: float(np.nanmean(values)) for group, values in groups.items()}
            for learned, groups in self._values.items()
        }
        held_out = [group for group in self._evaluated_groups if group not in self._learned_groups]

        return {
            f"{self._info_key()}_{self._base_metric.name()}": {
                "base_metric_name": self._base_metric.name(),
                "metrics": {m.name(): m.compute(self._square_matrix(group_matrix)) for m in self._summarized_metrics},
                "groups_order": list(self._learned_groups),
                "group_matrix": group_matrix,
                "held_out_groups": {
                    group: {learned: group_matrix[learned][group] for learned in self._learned_groups}
                    for group in held_out
                },
            }
        }

    def _concept_value(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        score_maps: Optional[np.ndarray],
    ) -> Optional[float]:
        return self._base_metric.compute(anomaly_scores=anomaly_scores, y_true=y_true, y_pred=y_pred)

    def _info_key(self) -> str:
        return "grouped_concept_metric_callback"

    def _square_matrix(self, group_matrix: Mapping[str, Mapping[str, float]]) -> ConceptLevelMatrix:
        if not self._learned_groups:
            return [[]]
        return [[group_matrix[learned][group] for group in self._learned_groups] for learned in self._learned_groups]
