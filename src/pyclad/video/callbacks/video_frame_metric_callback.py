"""Adapt frame curves to pyCLAD's existing concept-metric callback."""

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import SummarizedMetric
from pyclad.video.data.sample import VideoWindow
from pyclad.video.data.video_bag_concept import VideoBagConcept
from pyclad.video.metrics.frame_score_utils import (
    flatten_video_curves,
    window_scores_to_frame_scores,
)


class VideoFrameMetricCallback(ConceptMetricCallback):
    """Evaluate frame ranking metrics, reusing core matrix reporting.

    Skip when temporal predictions or frame labels are absent, as the vision
    pixel callback does when score maps or masks are unavailable.
    """

    def __init__(self, base_metric: BaseMetric, summarized_metrics: Iterable[SummarizedMetric] = ()):
        super().__init__(base_metric, summarized_metrics)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        anomaly_scores: Optional[np.ndarray] = None,
        window_scores: Optional[np.ndarray] = None,
        windows: Optional[Sequence[VideoWindow]] = None,
        *args,
        **kwargs,
    ) -> None:
        """Expand temporal scores against the evaluated concept's frame labels."""
        if (
            window_scores is None
            or windows is None
            or not isinstance(evaluated_concept, VideoBagConcept)
            or not evaluated_concept.frame_labels
        ):
            return
        labels = evaluated_concept.frame_labels
        scores = window_scores_to_frame_scores(
            windows, window_scores, {key: len(value) for key, value in labels.items()}
        )
        super().after_evaluation(
            evaluated_concept=evaluated_concept,
            y_true=flatten_video_curves(labels),
            y_pred=np.empty(0),
            anomaly_scores=flatten_video_curves(scores),
        )

    def info(self) -> Dict[str, Any]:
        if not self._evaluated_concepts:
            return {}
        return super().info()
