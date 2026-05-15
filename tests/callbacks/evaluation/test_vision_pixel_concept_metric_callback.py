import numpy as np

from pyclad.callbacks.evaluation.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.data.concept import Concept
from pyclad.data.vision_concept import VisionConcept
from pyclad.metrics.vision.pixel_f1_score import PixelF1Score
from pyclad.metrics.vision.pixel_roc_auc import PixelRocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage


class _ModelWithMaps:
    def score_maps(self, data: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [[0.95, 0.05], [0.05, 0.05]],
                [[0.05, 0.05], [0.05, 0.05]],
            ],
            dtype=np.float32,
        )


class _StrategyStub:
    def __init__(self, model):
        self._model = model


def test_vision_pixel_concept_metric_callback_computes_per_concept_metric():
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    concept = VisionConcept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
        masks=masks,
    )

    callback = VisionPixelConceptMetricCallback(
        strategy=_StrategyStub(_ModelWithMaps()),
        base_metrics=[PixelRocAuc()],
        summarized_metrics=[ContinualAverage()],
    )

    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
    )

    info = callback.info()["pixel_concept_metric_callback_Pixel-ROC-AUC"]
    assert info["evaluation_level"] == "pixel"
    assert info["concepts_order"] == ["widget"]
    assert info["metric_matrix"]["widget"]["widget"] == 1.0


def test_vision_pixel_concept_metric_callback_supports_multiple_base_metrics():
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    concept = VisionConcept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
        masks=masks,
    )

    callback = VisionPixelConceptMetricCallback(
        strategy=_StrategyStub(_ModelWithMaps()),
        base_metrics=[PixelRocAuc(), PixelF1Score(threshold=0.5)],
        summarized_metrics=[ContinualAverage()],
    )

    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
    )

    info = callback.info()
    assert info["pixel_concept_metric_callback_Pixel-ROC-AUC"]["metric_matrix"]["widget"]["widget"] == 1.0
    assert info["pixel_concept_metric_callback_Pixel-F1-Score"]["metric_matrix"]["widget"]["widget"] == 1.0


def test_vision_pixel_callback_skips_when_no_masks():
    concept = Concept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
    )

    callback = VisionPixelConceptMetricCallback(
        strategy=_StrategyStub(_ModelWithMaps()),
        base_metrics=[PixelRocAuc()],
    )

    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
    )

    # No pixel data → empty info
    info = callback.info()
    assert info == {}
