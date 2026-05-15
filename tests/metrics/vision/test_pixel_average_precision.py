import numpy as np

from pyclad.metrics.vision.pixel_average_precision import PixelAveragePrecision


def test_pixel_ap_perfect_localization():
    metric = PixelAveragePrecision()
    score_maps = np.array([[[0.9, 0.1], [0.8, 0.05]]], dtype=np.float32)
    masks = np.array([[[1, 0], [1, 0]]], dtype=np.uint8)
    assert metric.compute(anomaly_scores=score_maps, y_pred=np.array([]), y_true=masks) == 1.0


def test_pixel_ap_single_class_returns_nan():
    metric = PixelAveragePrecision()
    score_maps = np.array([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    masks = np.zeros((1, 2, 2), dtype=np.uint8)
    assert np.isnan(metric.compute(anomaly_scores=score_maps, y_pred=np.array([]), y_true=masks))


def test_pixel_ap_name():
    assert PixelAveragePrecision().name() == "Pixel-AP"
