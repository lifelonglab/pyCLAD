"""Frame-level metrics for video anomaly detection."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from pyclad.video.data.sample import VideoWindow


def window_scores_to_frame_scores(
    windows: Sequence[VideoWindow],
    window_scores: Sequence[float],
    frame_counts: Mapping[str, int],
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    """Aggregate temporal-window scores, requiring coverage of every labeled frame."""

    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be one of: 'mean', 'max'")
    if len(windows) != len(window_scores):
        raise ValueError("windows and window_scores must have the same length")

    initial = -np.inf if aggregation == "max" else 0.0
    scores = {
        video_id: np.full(frame_count, initial, dtype=np.float64) for video_id, frame_count in frame_counts.items()
    }
    counts = {video_id: np.zeros(frame_count, dtype=np.float64) for video_id, frame_count in frame_counts.items()}
    for video_id, frame_count in frame_counts.items():
        if frame_count <= 0:
            raise ValueError(f"frame count must be positive for video_id={video_id!r}")

    for window, score in zip(windows, window_scores):
        if window.video_id not in frame_counts:
            raise KeyError(f"Missing frame count for video_id={window.video_id!r}")
        frame_count = frame_counts[window.video_id]
        if window.start_frame >= frame_count:
            raise ValueError(
                f"Window starts outside video_id={window.video_id!r}: " f"start_frame={window.start_frame}"
            )
        frame_slice = slice(window.start_frame, min(window.end_frame, frame_count - 1) + 1)
        if aggregation == "max":
            scores[window.video_id][frame_slice] = np.maximum(
                scores[window.video_id][frame_slice],
                float(score),
            )
        else:
            scores[window.video_id][frame_slice] += float(score)
        counts[window.video_id][frame_slice] += 1.0

    for video_id, values in scores.items():
        covered = counts[video_id] > 0
        if not covered.all():
            raise ValueError(f"window scores do not cover all frames for video_id={video_id!r}")
        if aggregation == "mean":
            values[covered] /= counts[video_id][covered]
    return scores


def flatten_video_curves(curves_by_video: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenate video curves in stable video-id order."""

    if not curves_by_video:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([np.asarray(curves_by_video[video_id]).reshape(-1) for video_id in sorted(curves_by_video)])
