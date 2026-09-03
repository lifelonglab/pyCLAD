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
from pyclad.vision.metrics.pixel_dice_score import PixelDiceScore
from pyclad.vision.metrics.pixel_f1_score import PixelF1Score
from pyclad.vision.metrics.pixel_iou import PixelIoU
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.patchcore.config import PatchCoreConfig
from pyclad.vision.models.patchcore.patchcore import PatchCore
from pyclad.vision.strategies.replaycad.backend import DiffusersBackend
from pyclad.vision.strategies.replaycad.config import ReplayCADConfig
from pyclad.vision.strategies.replaycad.memory import ReplayCADMemory
from pyclad.vision.strategies.replaycad.strategy import ReplayCADStrategy

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """
    This example showcases ReplayCAD -- diffusion-based generative replay -- with PatchCore as the
    (detector-agnostic) anomaly detector. It uses ConceptAwareScenario, not
    ConceptIncrementalScenario, since ReplayCAD needs the concept id on every call.

    Requires the 'replaycad' extra (pip install -e ".[replaycad]") and either a SAM checkpoint
    (mask_backend="sam") or the authors' precomputed SAM.zip masks (mask_backend="precomputed",
    used below). See docs/vision.md for mask backends and the divergences from the paper.
    """
    dataset = read_vision_dataset(
        root=pathlib.Path("../resources/vision/mvtec_ad"),
        benchmark="mvtec",
        resize_to=(256, 256),
        data_mode="numpy",
        color_mode="rgb",
        # max_train_samples_per_category=150,
        # max_test_samples_per_category=150,
    )

    model = PatchCore(
        PatchCoreConfig(
            input_size=(256, 256),
            backbone_name="wide_resnet50_2",
            pretrained_backbone=True,
            pretrained_weights="IMAGENET1K_V1",
            batch_size=2,
            seed=42,
        )
    )

    replaycad_config = ReplayCADConfig.for_benchmark(
        "mvtec",
        artifact_dir=pathlib.Path("./replaycad-artifacts"),
        mask_backend="precomputed",
        precomputed_mask_root=pathlib.Path("./SAM/data"),
        # The authors' SAM.zip has no masks for "zipper"; route it to the full-frame fallback
        # instead of letting PrecomputedMaskProvider raise on it.
        mask_modes={"zipper": "full-frame"},
        device="cuda",
        seed=42,
    )
    memory = ReplayCADMemory(
        config=replaycad_config,
        backend=DiffusersBackend(replaycad_config),
        benchmark="mvtec",
    )
    strategy = ReplayCADStrategy(model, memory)

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

    scenario = ConceptAwareScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
