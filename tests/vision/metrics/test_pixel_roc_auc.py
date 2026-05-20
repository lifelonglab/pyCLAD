import numpy as np

from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc


def test_pixel_roc_auc_flattens_pixel_maps():
    metric = PixelRocAuc()

    score_maps = np.array(
        [
            [[0.1, 0.2], [0.9, 0.8]],
            [[0.3, 0.4], [0.7, 0.6]],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )

    value = metric.compute(anomaly_scores=score_maps, y_pred=np.asarray([]), y_true=masks)

    assert value == 1.0
