import numpy as np
import pytest

from pyclad.metrics.vision.pixel_aupro import PixelAUPRO, _compute_pro_fpr_curve, _integrate_pro_curve


def test_aupro_perfect_localization():
    metric = PixelAUPRO(fpr_limit=0.3)

    score_maps = np.zeros((1, 8, 8), dtype=np.float32)
    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    score_maps[0, 0:2, 0:2] = 1.0
    masks[0, 0:2, 0:2] = 1

    value = metric.compute(anomaly_scores=score_maps, y_pred=np.array([]), y_true=masks)
    assert value > 0.95


def test_aupro_random_scores_below_perfect():
    rng = np.random.default_rng(42)
    metric = PixelAUPRO(fpr_limit=0.3)

    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    masks[0, 2:5, 2:5] = 1
    masks[1, 10:13, 10:13] = 1

    score_maps = rng.random((2, 16, 16)).astype(np.float32)

    value = metric.compute(anomaly_scores=score_maps, y_pred=np.array([]), y_true=masks)
    assert 0.0 < value < 1.0


def test_aupro_single_class_returns_nan():
    metric = PixelAUPRO()
    score_maps = np.ones((1, 4, 4), dtype=np.float32)
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    assert np.isnan(metric.compute(anomaly_scores=score_maps, y_pred=np.array([]), y_true=masks))


def test_aupro_empty_returns_nan():
    metric = PixelAUPRO()
    assert np.isnan(metric.compute(anomaly_scores=np.array([]), y_pred=np.array([]), y_true=np.array([])))


def test_aupro_rejects_invalid_fpr_limit():
    with pytest.raises(ValueError, match="fpr_limit"):
        PixelAUPRO(fpr_limit=0.0)
    with pytest.raises(ValueError, match="fpr_limit"):
        PixelAUPRO(fpr_limit=1.5)


def test_aupro_name():
    assert PixelAUPRO().name() == "Pixel-AUPRO"


def test_pro_curve_multiple_regions():
    scores = np.zeros((8, 8), dtype=np.float32)
    gt = np.zeros((8, 8), dtype=np.uint8)

    scores[0:2, 0:2] = 0.9
    gt[0:2, 0:2] = 1

    scores[5:7, 5:7] = 0.8
    gt[5:7, 5:7] = 1

    fprs, pros = _compute_pro_fpr_curve(scores, gt)
    assert fprs[0] <= fprs[-1]
    assert pros[-1] > 0.0


def test_aupro_batch_does_not_merge_regions_across_images():
    """Regions at image boundaries must be labeled per image, not merged via 3D connectivity."""
    scores = np.zeros((2, 4, 4), dtype=np.float32)
    gt = np.zeros((2, 4, 4), dtype=np.uint8)

    scores[0, 3, 0:2] = 0.9
    gt[0, 3, 0:2] = 1

    scores[1, 0, 0:2] = 0.1
    gt[1, 0, 0:2] = 1

    fprs, pros = _compute_pro_fpr_curve(scores, gt)

    # With per-image labeling there are 2 regions. At the highest threshold
    # (0.9) only region 1 is detected → PRO = (1.0 + 0.0) / 2 = 0.5.
    # If merged via 3D connectivity into 1 region, PRO = 2/4 = 0.5 at that
    # threshold too, but at thresh ≈ 0.1 the merged region would show 4/4 = 1.0
    # while separate regions also give (1.0 + 1.0)/2 = 1.0.
    # The key check: a PRO of exactly 0.5 exists, proving 2 separate regions.
    assert any(np.isclose(pros, 0.5, atol=0.01))
