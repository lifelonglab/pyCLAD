from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResults:
    y_pred: np.ndarray
    anomaly_scores: np.ndarray