import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.callbacks.evaluation.vision_pixel_concept_metric_callback import VisionPixelConceptMetricCallback
from pyclad.data.readers.vision_reader import read_vision_dataset
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.metrics.vision.pixel_aupro import PixelAUPRO
from pyclad.metrics.vision.pixel_average_precision import PixelAveragePrecision
from pyclad.metrics.vision.pixel_dice_score import PixelDiceScore
from pyclad.metrics.vision.pixel_f1_score import PixelF1Score
from pyclad.metrics.vision.pixel_iou import PixelIoU
from pyclad.metrics.vision.pixel_roc_auc import PixelRocAuc
from pyclad.models.vision.paste.config import PaSTeConfig
from pyclad.models.vision.paste.paste import PaSTe
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example showcases how to use the PaSTe model for continual vision anomaly detection.
    """
    dataset = read_vision_dataset(
        benchmark="btech",
        resize_to=(224, 224),
        data_mode="numpy",
        color_mode="rgb",
        max_train_samples_per_category=150,
        max_test_samples_per_category=150,
    )

    model_config = PaSTeConfig(
        input_size=(224, 224),
        backbone_name="mobilenet_v2",
        student_bootstrap_layer=0,
        pretrained_teacher=True,
        pretrained_student=False,
        batch_size=16,
        epochs=50,
        score_mode="max",
        threshold_quantile=0.99,
        show_training_progress=True,
    )
    model = PaSTe(model_config)

    replay_buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=100)
    strategy = ReplayEnhancedStrategy(model, replay_buffer)

    summarized_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer()]

    callbacks = [
        # Image-level
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=F1Score(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=summarized_metrics),
        # Pixel-level
        VisionPixelConceptMetricCallback(
            strategy=strategy,
            base_metrics=[PixelRocAuc(), PixelAveragePrecision(), PixelAUPRO(), PixelF1Score(), PixelDiceScore(),
                          PixelIoU()],
            summarized_metrics=summarized_metrics,
        ),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("paste_output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
