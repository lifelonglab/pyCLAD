import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.readers.vision_reader import read_vision_dataset
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.metrics.pixel_dice_score import PixelDiceScore
from pyclad.vision.metrics.pixel_f1_score import PixelF1Score
from pyclad.vision.metrics.pixel_iou import PixelIoU
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.stfpm.config import STFPMConfig
from pyclad.vision.models.stfpm.stfpm import STFPM

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """
    This example showcases how to use the STFPM model for continual vision anomaly detection.
    """
    dataset = read_vision_dataset(
        root=pathlib.Path("../../resources/vision/BTech_Dataset_transformed"),
        benchmark="btech",
        resize_to=(256, 256),
        data_mode="numpy",
        color_mode="rgb",
        # max_train_samples_per_category=150,
        # max_test_samples_per_category=150,
    )

    model_config = STFPMConfig(
        input_size=(256, 256),
        backbone_name="resnet18",
        pretrained_teacher=True,
        pretrained_student=False,
        freeze_teacher=True,
        batch_size=32,
        epochs=2,
        learning_rate=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        score_mode="max",
        threshold_quantile=0.99,
        show_training_progress=True,
    )
    model = STFPM(model_config)

    replay_buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=100)
    strategy = ReplayEnhancedStrategy(model, replay_buffer)

    summarized_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer()]

    callbacks = [
        # Image-level
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=F1Score(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=summarized_metrics),
        # Pixel-level
        VisionPixelConceptMetricCallback(base_metric=PixelRocAuc(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelAveragePrecision(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelAUPRO(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelF1Score(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelDiceScore(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelIoU(), summarized_metrics=summarized_metrics),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
