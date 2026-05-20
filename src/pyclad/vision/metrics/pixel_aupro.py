from __future__ import annotations

import numpy as np
from scipy import ndimage

from pyclad.metrics.base.base_metric import BaseMetric


def _compute_pro_fpr_curve(
    anomaly_scores: np.ndarray,
    ground_truth: np.ndarray,
    num_thresholds: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the PRO (Per-Region Overlap) vs FPR curve.

    For each threshold, binarize the score map and compute:
    - FPR: false positive rate over all normal pixels
    - PRO: mean per-region overlap (TPR averaged across connected components)

    Connected components are labeled per image to avoid merging
    regions across the batch dimension.
    """
    if anomaly_scores.ndim == 2:
        anomaly_scores = anomaly_scores[np.newaxis]
        ground_truth = ground_truth[np.newaxis]

    scores_flat = anomaly_scores.reshape(-1).astype(np.float32)
    gt_flat = ground_truth.reshape(-1).astype(np.uint8)

    normal_mask = gt_flat == 0
    num_normal = int(normal_mask.sum())

    region_pixels: list[np.ndarray] = []
    for img_scores, img_gt in zip(anomaly_scores, ground_truth):
        labeled, num_regions = ndimage.label(img_gt)
        for region_id in range(1, num_regions + 1):
            region_pixels.append(img_scores[labeled == region_id].astype(np.float32))

    if len(region_pixels) == 0 or num_normal == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    thresholds = np.percentile(scores_flat, np.linspace(100, 0, num_thresholds))
    thresholds = np.unique(thresholds)

    fprs = np.zeros(len(thresholds))
    pros = np.zeros(len(thresholds))

    for i, thresh in enumerate(thresholds):
        binary = (scores_flat >= thresh).astype(np.uint8)
        fprs[i] = binary[normal_mask].sum() / num_normal

        region_overlaps = np.array([(pixels >= thresh).mean() for pixels in region_pixels])
        pros[i] = region_overlaps.mean()

    sort_idx = np.argsort(fprs)
    return fprs[sort_idx], pros[sort_idx]


def _integrate_pro_curve(fprs: np.ndarray, pros: np.ndarray, fpr_limit: float) -> float:
    """Integrate the PRO curve up to fpr_limit and normalize to [0, 1]."""
    mask = fprs <= fpr_limit

    if mask.sum() < 2:
        return 0.0

    fprs_clipped = fprs[mask]
    pros_clipped = pros[mask]

    if fprs_clipped[-1] < fpr_limit:
        pros_clipped = np.append(pros_clipped, pros_clipped[-1])
        fprs_clipped = np.append(fprs_clipped, fpr_limit)

    area = float(np.trapezoid(pros_clipped, fprs_clipped))
    return area / fpr_limit


class PixelAUPRO(BaseMetric):
    """Area Under the Per-Region Overlap curve.

    Standard localization metric for industrial anomaly detection (MVTec).
    Measures how well the model localizes each individual anomalous region,
    normalized by false positive rate up to ``fpr_limit``.
    """

    def __init__(self, fpr_limit: float = 0.3, num_thresholds: int = 300):
        if not 0.0 < fpr_limit <= 1.0:
            raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")
        self.fpr_limit = float(fpr_limit)
        self.num_thresholds = int(num_thresholds)

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        scores = np.asarray(anomaly_scores, dtype=np.float32)
        gt = np.asarray(y_true, dtype=np.uint8)

        if scores.size == 0 or gt.size == 0:
            return float("nan")
        if len(np.unique(gt)) < 2:
            return float("nan")

        fprs, pros = _compute_pro_fpr_curve(scores, gt, self.num_thresholds)
        return _integrate_pro_curve(fprs, pros, self.fpr_limit)

    def name(self) -> str:
        return "Pixel-AUPRO"
