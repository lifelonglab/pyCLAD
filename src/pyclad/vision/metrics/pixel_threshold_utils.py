from __future__ import annotations

import numpy as np


def threshold_pixel_scores(anomaly_scores, threshold: float) -> np.ndarray:
    return (np.asarray(anomaly_scores) > float(threshold)).astype(np.uint8, copy=False)


def flatten_binary_masks(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint8).reshape(-1)


def flatten_pixel_scores(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def resolve_pixel_threshold(
    anomaly_scores,
    *,
    mode: str = "fixed",
    fixed_threshold: float = 0.5,
    quantile: float = 0.995,
) -> float:
    if mode == "fixed":
        return float(fixed_threshold)

    if mode != "train-quantile":
        raise ValueError(f"Unsupported pixel threshold mode: {mode}")

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"Pixel threshold quantile must be in (0, 1), got {quantile}")

    scores_flat = flatten_pixel_scores(anomaly_scores)
    if scores_flat.size == 0:
        return float(fixed_threshold)
    return float(np.quantile(scores_flat, float(quantile)))


def binary_confusion(y_true, y_pred) -> tuple[int, int, int]:
    y_true_flat = flatten_binary_masks(y_true)
    y_pred_flat = flatten_binary_masks(y_pred)

    tp = int(np.sum((y_true_flat == 1) & (y_pred_flat == 1)))
    fp = int(np.sum((y_true_flat == 0) & (y_pred_flat == 1)))
    fn = int(np.sum((y_true_flat == 1) & (y_pred_flat == 0)))
    return tp, fp, fn
