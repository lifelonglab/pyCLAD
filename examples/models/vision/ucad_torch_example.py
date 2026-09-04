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
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.readers.vision_reader import read_vision_dataset
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.ucad import UCAD
from pyclad.vision.strategies.ucad.strategy import UCADStrategy

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """
    UCAD (Liu et al., AAAI 2024) for continual visual anomaly detection.

    NOTE: results from this port are NOT comparable to the paper's Tables 1-4. The reference
    implementation selects the prompt and knowledge bank by the best test-set AUROC over 25
    epochs and never runs its continual evaluation loop. See docs/vision.md.
    """
    dataset = read_vision_dataset(
        root=pathlib.Path("../../resources/vision/BTech_Dataset_transformed"),
        benchmark="btech",
        resize_to=(224, 224),
        data_mode="numpy",
        color_mode="rgb",
    )

    model_config = UCADConfig(
        input_size=(224, 224),
        batch_size=8,
        epochs=25,
        # CPM-only by default. For the full CPM+SCL method, point these at the authors'
        # precomputed SAM maps (the mvtec2d-sam-b archive):
        #   structure_mode="precomputed",
        #   structure_mask_root="/data/mvtec2d-sam-b",
        structure_mode="none",
        seed=42,
    )
    model = UCAD(model_config)

    # No replay wrapper: UCAD's continual mechanism is its own key-prompt-knowledge memory.
    strategy = UCADStrategy(model)

    summarized_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer()]

    callbacks = [
        # Image-level
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=F1Score(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelRocAuc(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelAveragePrecision(), summarized_metrics=summarized_metrics),
        VisionPixelConceptMetricCallback(base_metric=PixelAUPRO(), summarized_metrics=summarized_metrics),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptAwareScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
