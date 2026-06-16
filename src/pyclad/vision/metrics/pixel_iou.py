from pyclad.vision.metrics.pixel_metric import PixelMetric
from pyclad.vision.metrics.pixel_threshold_utils import binary_confusion


class PixelIoU(PixelMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_pred_binary = self._binarize(anomaly_scores)
        tp, fp, fn = binary_confusion(y_true, y_pred_binary)

        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0
        return float(tp / denominator)

    def name(self) -> str:
        return "Pixel-IoU"
