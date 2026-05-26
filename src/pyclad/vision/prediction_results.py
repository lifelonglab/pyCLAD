from dataclasses import dataclass

import numpy as np

from pyclad.output.prediction_results import PredictionResults


@dataclass
class VisionPredictionResults(PredictionResults):
    score_maps: np.ndarray