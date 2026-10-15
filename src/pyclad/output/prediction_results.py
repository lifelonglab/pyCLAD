from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResults:
    """What a model or a strategy returns from ``predict``, one entry per sample.

    :param y_pred: predicted labels, ``0`` for a normal sample and ``1`` for an anomaly.
    :param anomaly_scores: continuous scores, where a higher score means a more anomalous sample. Their
        scale is up to the model; ranking metrics such as ROC-AUC only use their order.
    """

    y_pred: np.ndarray
    anomaly_scores: np.ndarray
