from abc import abstractmethod

import numpy as np

from pyclad.models.model import Model


class VisionModel(Model):
    """Anomaly detection model producing pixel-level score maps."""

    @abstractmethod
    def score_maps(self, data: np.ndarray) -> np.ndarray:
        """Compute per-pixel anomaly scores.

        :param data: batch of images, shape ``(N, H, W, C)``.
        :return: array of shape ``(N, H, W)`` — higher values are more anomalous.
        """
        ...
