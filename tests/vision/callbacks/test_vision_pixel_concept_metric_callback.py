import numpy as np

from pyclad.data.concept import Concept
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc


def _make_vision_concept(name: str = "widget") -> VisionConcept:
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    return VisionConcept(
        name=name,
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
        masks=masks,
    )


def _score_maps_predicting_first_pixel_as_anomaly() -> np.ndarray:
    return np.array(
        [
            [[0.95, 0.05], [0.05, 0.05]],
            [[0.05, 0.05], [0.05, 0.05]],
        ],
        dtype=np.float32,
    )


def test_callback_records_pixel_metric_when_scenario_provides_score_maps():
    """In Opcja-2 architecture, the vision scenario passes score_maps via
    after_evaluation kwarg. Callback must NOT touch strategy/model itself."""
    callback = VisionPixelConceptMetricCallback(
        base_metric=PixelRocAuc(),
        summarized_metrics=[ContinualAverage()],
    )
    concept = _make_vision_concept()

    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
        score_maps=_score_maps_predicting_first_pixel_as_anomaly(),
    )

    info = callback.info()["pixel_concept_metric_callback_Pixel-ROC-AUC"]
    assert info["evaluation_level"] == "pixel"
    assert info["concepts_order"] == ["widget"]
    assert info["metric_matrix"]["widget"]["widget"] == 1.0


def test_callback_skips_when_score_maps_missing():
    """When running under a non-vision scenario, score_maps is absent → skip."""
    callback = VisionPixelConceptMetricCallback(base_metric=PixelRocAuc())
    concept = _make_vision_concept()

    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
        # no score_maps kwarg
    )

    assert callback.info() == {}


def test_callback_skips_when_concept_is_not_vision_concept():
    """Non-vision concept must be skipped even if score_maps somehow arrived."""
    callback = VisionPixelConceptMetricCallback(base_metric=PixelRocAuc())
    tabular_concept = Concept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
    )

    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=tabular_concept,
        y_true=tabular_concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
        score_maps=_score_maps_predicting_first_pixel_as_anomaly(),
    )

    assert callback.info() == {}


def test_callback_constructor_matches_concept_metric_callback_shape():
    """One base_metric per callback, optional summarized_metrics. No strategy.
    Same shape as :class:`ConceptMetricCallback`."""
    callback = VisionPixelConceptMetricCallback(
        base_metric=PixelRocAuc(),
        summarized_metrics=[ContinualAverage()],
    )

    assert callback._base_metric.name() == "Pixel-ROC-AUC"
    assert not hasattr(callback, "_strategy")
