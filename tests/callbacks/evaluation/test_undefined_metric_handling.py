import numpy as np
import pytest

from pyclad.callbacks import (
    ConceptMetricCallback,
    GroupedConceptMetricCallback,
    ScheduleAwareConceptMetricCallback,
    UndefinedMetricError,
)
from pyclad.data import Concept
from pyclad.metrics import ContinualAverage, RocAuc, ScheduleAwareNewTaskAcquisition
from pyclad.vision.callbacks.grouped_vision_pixel_concept_metric_callback import (
    GroupedVisionPixelConceptMetricCallback,
)
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    ScheduleAwareVisionPixelConceptMetricCallback,
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc

SCORES = np.array([0.1, 0.2, 0.8, 0.9])
LABELS = {"a": np.array([0, 0, 1, 1]), "b": np.zeros(4)}  # "b" has no anomalies, so ROC-AUC is undefined
WARNING = (
    "ROC-AUC is undefined for concept 'b' (for example, its test data holds a single class). "
    "Continual metrics that read this concept will be NaN."
)


def _run(callback):
    for learned in ["a", "b"]:
        callback.after_training(Concept(learned, data=np.array([])))
        for evaluated in ["a", "b"]:
            callback.after_evaluation(Concept(evaluated, data=np.array([])), LABELS[evaluated], np.array([]), SCORES)
    return next(iter(callback.info().values()))


def _concept_callback(**kwargs):
    return ConceptMetricCallback(RocAuc(), [ContinualAverage()], **kwargs)


def _schedule_aware_callback(**kwargs):
    return ScheduleAwareConceptMetricCallback(
        RocAuc(),
        [ContinualAverage()],
        schedule_aware_metrics=[ScheduleAwareNewTaskAcquisition()],
        first_seen_step={"a": 0, "b": 1},
        **kwargs,
    )


def _grouped_callback(**kwargs):
    return GroupedConceptMetricCallback(RocAuc(), {"a": "a", "b": "b"}, [ContinualAverage()], **kwargs)


CALLBACK_FACTORIES = [_concept_callback, _schedule_aware_callback, _grouped_callback]


@pytest.mark.parametrize("make_callback", CALLBACK_FACTORIES)
def test_undefined_base_metric_raises_by_default(make_callback):
    with pytest.raises(UndefinedMetricError, match="ROC-AUC is undefined for concept 'b'.*on_undefined='propagate'"):
        _run(make_callback())


@pytest.mark.parametrize("make_callback", CALLBACK_FACTORIES)
def test_propagate_mode_reports_the_undefined_concept_once(make_callback, caplog):
    with caplog.at_level("WARNING"):
        info = _run(make_callback(on_undefined="propagate"))

    assert info["undefined_concepts"] == ["b"]
    assert [record.getMessage() for record in caplog.records].count(WARNING) == 1


def test_propagate_mode_keeps_the_matrix_and_reports_nan_for_metrics_that_read_the_concept():
    info = _run(_concept_callback(on_undefined="propagate"))

    assert info["metric_matrix"]["a"]["a"] == pytest.approx(1.0)
    assert np.isnan(info["metric_matrix"]["a"]["b"])
    assert np.isnan(info["metrics"]["ContinualAverage"])


def test_propagate_mode_averages_a_group_over_its_defined_concepts():
    callback = GroupedConceptMetricCallback(RocAuc(), {"a": "group", "b": "group"}, on_undefined="propagate")
    callback.after_training(Concept("group", data=np.array([])))
    for evaluated in ["a", "b"]:
        callback.after_evaluation(Concept(evaluated, data=np.array([])), LABELS[evaluated], np.array([]), SCORES)

    info = next(iter(callback.info().values()))

    assert info["group_matrix"]["group"]["group"] == pytest.approx(1.0)  # concept "b" is left out of the average
    assert info["undefined_concepts"] == ["b"]


def test_propagate_mode_reports_nan_for_a_group_with_no_defined_concept():
    info = _run(_grouped_callback(on_undefined="propagate"))  # group "b" holds concept "b" only

    assert np.isnan(info["group_matrix"]["a"]["b"])
    assert np.isnan(info["metrics"]["ContinualAverage"])


def test_nothing_is_reported_when_every_concept_is_defined():
    callback = _concept_callback()
    callback.after_training(Concept("a", data=np.array([])))
    callback.after_evaluation(Concept("a", data=np.array([])), LABELS["a"], np.array([]), SCORES)

    assert callback.info()["concept_metric_callback_ROC-AUC"]["undefined_concepts"] == []


@pytest.mark.parametrize(
    "make_callback",
    [
        *CALLBACK_FACTORIES,
        lambda **kwargs: VisionPixelConceptMetricCallback(PixelRocAuc(), **kwargs),
        lambda **kwargs: GroupedVisionPixelConceptMetricCallback(PixelRocAuc(), {}, **kwargs),
    ],
)
def test_unknown_on_undefined_option_is_rejected(make_callback):
    with pytest.raises(ValueError, match="on_undefined must be one of"):
        make_callback(on_undefined="ignore")


def _concept_without_anomalous_pixels() -> VisionConcept:
    return VisionConcept(
        name="widget",
        data=np.zeros((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([0, 0], dtype=np.int64),
        masks=np.zeros((2, 2, 2), dtype=np.uint8),
    )


def _pixel_callback(**kwargs):
    return VisionPixelConceptMetricCallback(PixelRocAuc(), **kwargs)


def _schedule_aware_pixel_callback(**kwargs):
    return ScheduleAwareVisionPixelConceptMetricCallback(
        PixelRocAuc(),
        schedule_aware_metrics=[ScheduleAwareNewTaskAcquisition()],
        first_seen_step={"widget": 0},
        **kwargs,
    )


def _grouped_pixel_callback(**kwargs):
    return GroupedVisionPixelConceptMetricCallback(PixelRocAuc(), {"widget": "widgets"}, **kwargs)


PIXEL_CALLBACK_FACTORIES = [_pixel_callback, _schedule_aware_pixel_callback, _grouped_pixel_callback]


def _run_pixel(callback):
    concept = _concept_without_anomalous_pixels()
    callback.after_training(Concept(name="widget", data=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([0, 0]),
        anomaly_scores=np.array([0.1, 0.2]),
        score_maps=np.full((2, 2, 2), 0.1, dtype=np.float32),
    )
    return next(iter(callback.info().values()))


@pytest.mark.parametrize("make_callback", PIXEL_CALLBACK_FACTORIES)
def test_undefined_pixel_metric_raises_by_default(make_callback):
    with pytest.raises(UndefinedMetricError, match="Pixel-ROC-AUC is undefined for concept 'widget'"):
        _run_pixel(make_callback())


@pytest.mark.parametrize("make_callback", PIXEL_CALLBACK_FACTORIES)
def test_propagate_mode_reports_the_undefined_pixel_concept(make_callback):
    assert _run_pixel(make_callback(on_undefined="propagate"))["undefined_concepts"] == ["widget"]
