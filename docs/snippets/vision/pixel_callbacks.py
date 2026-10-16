# requires: torch, torchvision
# --8<-- [start:callbacks]
from pyclad.callbacks import ConceptMetricCallback
from pyclad.metrics import AveragePrecision, BackwardTransfer, ContinualAverage, F1Score, ForwardTransfer, RocAuc
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import VisionPixelConceptMetricCallback
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc

summarized_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer()]

callbacks = [
    # image-level
    ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics),
    ConceptMetricCallback(base_metric=F1Score(), summarized_metrics=summarized_metrics),
    ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=summarized_metrics),
    # pixel-level (need masks)
    VisionPixelConceptMetricCallback(base_metric=PixelRocAuc(), summarized_metrics=summarized_metrics),
    VisionPixelConceptMetricCallback(base_metric=PixelAUPRO(), summarized_metrics=summarized_metrics),
    # ... PixelAveragePrecision, PixelF1Score, PixelIoU, PixelDiceScore
]
# --8<-- [end:callbacks]
