import numpy as np

from pyclad.metrics.vision.pixel_dice_score import PixelDiceScore
from pyclad.metrics.vision.pixel_f1_score import PixelF1Score
from pyclad.metrics.vision.pixel_iou import PixelIoU
from pyclad.metrics.vision.pixel_threshold_utils import resolve_pixel_threshold


def test_thresholded_pixel_metrics_compute_expected_scores():
    score_maps = np.array([[[0.9, 0.2], [0.8, 0.1]]], dtype=np.float32)
    masks = np.array([[[1, 0], [1, 0]]], dtype=np.uint8)

    f1 = PixelF1Score(threshold=0.5).compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)
    dice = PixelDiceScore(threshold=0.5).compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)
    iou = PixelIoU(threshold=0.5).compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)

    assert f1 == 1.0
    assert dice == 1.0
    assert iou == 1.0


def test_thresholded_pixel_metrics_handle_empty_masks_and_predictions():
    score_maps = np.zeros((1, 2, 2), dtype=np.float32)
    masks = np.zeros((1, 2, 2), dtype=np.uint8)

    dice = PixelDiceScore(threshold=0.5).compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)
    iou = PixelIoU(threshold=0.5).compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)

    assert dice == 1.0
    assert iou == 1.0


def test_thresholded_pixel_metrics_support_runtime_threshold_override():
    score_maps = np.array([[[0.9, 0.2], [0.8, 0.1]]], dtype=np.float32)
    masks = np.array([[[1, 0], [1, 0]]], dtype=np.uint8)

    metric = PixelF1Score(threshold=0.95)
    metric.set_runtime_threshold(0.5)

    assert metric.compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks) == 1.0


def test_resolve_pixel_threshold_supports_train_quantile_mode():
    train_scores = np.array([0.01, 0.02, 0.05, 0.2], dtype=np.float32)
    threshold = resolve_pixel_threshold(
        train_scores,
        mode="train-quantile",
        fixed_threshold=0.5,
        quantile=0.75,
    )

    assert np.isclose(threshold, 0.0875)
