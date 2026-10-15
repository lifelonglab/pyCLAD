"""Bag predictions with temporal scores for frame-level evaluation."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pyclad.output.prediction_results import PredictionResults
from pyclad.video.data.sample import VideoWindow


@dataclass
class VideoPredictionResults(PredictionResults):
    """Bag-level maxima plus aligned window scores and metadata."""

    window_scores: np.ndarray
    classifier_scores: np.ndarray
    windows: Tuple[VideoWindow, ...]
