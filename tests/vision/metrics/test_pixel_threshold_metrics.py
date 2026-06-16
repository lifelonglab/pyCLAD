import numpy as np

from pyclad.vision.metrics.pixel_dice_score import PixelDiceScore
from pyclad.vision.metrics.pixel_f1_score import PixelF1Score
from pyclad.vision.metrics.pixel_iou import PixelIoU
from pyclad.vision.metrics.pixel_threshold_utils import resolve_pixel_threshold


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


def test_resolve_pixel_threshold_supports_train_quantile_mode():
    train_scores = np.array([0.01, 0.02, 0.05, 0.2], dtype=np.float32)
    threshold = resolve_pixel_threshold(
        train_scores,
        mode="train-quantile",
        fixed_threshold=0.5,
        quantile=0.75,
    )

    assert np.isclose(threshold, 0.0875)


def test_default_pixel_threshold_is_scale_invariant_and_non_degenerate():
    # Same relative anomaly pattern, two absolute scales: positive (PaSTe-like)
    # and shifted into the negative range (FastFlow maps live in [-1, 0)).
    rng = np.random.default_rng(0)
    score_maps = rng.uniform(0.0, 0.2, size=(1, 10, 10)).astype(np.float32)
    score_maps[0, 0, :5] = rng.uniform(0.8, 1.0, size=5)  # 5 distinct anomalous pixels (highest)
    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 0, :5] = 1

    metric = PixelF1Score()  # data-driven (quantile) threshold by default
    f1_positive = metric.compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)
    f1_negative = metric.compute(anomaly_scores=score_maps - 1.0, y_pred=np.asarray([]), y_true=masks)

    # The quantile threshold shifts with the data, so the binarization — and the
    # score — is identical regardless of the absolute scale of the maps.
    assert f1_positive == f1_negative
    assert f1_positive > 0.0

    # The old fixed-0.5 default silently collapses to 0 on FastFlow-scale maps.
    f1_fixed = PixelF1Score(threshold=0.5).compute(anomaly_scores=score_maps - 1.0, y_pred=np.asarray([]), y_true=masks)
    assert f1_fixed == 0.0
