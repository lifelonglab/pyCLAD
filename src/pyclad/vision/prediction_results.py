from dataclasses import dataclass

import numpy as np

from pyclad.output.prediction_results import PredictionResults


@dataclass
class VisionPredictionResults(PredictionResults):
    """Image-level predictions, as in :class:`PredictionResults`, with per-pixel scores added.

    :param score_maps: one anomaly score per pixel of every image, where a higher score means a more
        anomalous pixel.
    """

    score_maps: np.ndarray
