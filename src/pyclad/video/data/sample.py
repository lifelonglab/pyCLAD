"""Complete video bags and their temporal annotation coordinates."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VideoWindow:
    """Inclusive frame coordinates for one temporal snippet."""

    video_id: str
    start_frame: int
    end_frame: int

    def __post_init__(self):
        if not self.video_id or self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("a window requires a video id and nonnegative ordered frame coordinates")


@dataclass(frozen=True)
class VideoBag:
    """One complete video bag retained by the paper trainer and replay buffer."""

    bag_id: str
    task_id: str
    features: np.ndarray
    weak_label: int
    windows: Tuple[VideoWindow, ...] = ()

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if not self.bag_id or not self.task_id:
            raise ValueError("bag_id and task_id must be non-empty")
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(f"paper COMMAND bags must have shape (time, features), got {features.shape}")
        if len(features) == 0 or not np.isfinite(features).all():
            raise ValueError("paper COMMAND bag features must be finite and non-empty")
        if self.weak_label not in {0, 1}:
            raise ValueError("weak_label must be zero or one")
        if self.windows and len(self.windows) != len(features):
            raise ValueError("windows and bag features must have the same temporal length")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "windows", tuple(self.windows))


def as_video_bags(data: np.ndarray | Sequence[VideoBag]) -> Tuple[VideoBag, ...]:
    """Validate the object-array rows shared by video models and strategies."""
    values = np.asarray(data, dtype=object)
    if values.ndim != 1 or any(not isinstance(bag, VideoBag) for bag in values):
        raise ValueError("COMMAND requires a one-dimensional array of VideoBag objects")
    return tuple(values)
