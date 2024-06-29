import numpy as np


def adjust_predictions(predictions):
    predictions = np.where(predictions == 1, 0, predictions)
    predictions = np.where(predictions == -1, 1, predictions)
    return predictions
