import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.grouping import apply_step_schedule
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.metrics.continual.schedule_aware_forgetting_measure import (
    ScheduleAwareForgettingMeasure,
)
from pyclad.metrics.continual.schedule_aware_forward_transfer import (
    ScheduleAwareForwardTransfer,
)
from pyclad.metrics.continual.schedule_aware_new_task_acquisition import (
    ScheduleAwareNewTaskAcquisition,
)
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.multi_dataset import (
    load_inclad_md,
    split_multi_dataset_concept_name,
)
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.rd4ad.rd4ad import RD4AD

logging.basicConfig(level=logging.INFO)

RESOURCES = pathlib.Path("resources/vision")
ROOTS = {
    "btech": RESOURCES / "BTech_Dataset_transformed",
    "dagm": RESOURCES / "DAGM_KaggleUpload",
    "mpdd": RESOURCES / "MPDD",
    "mvtec": RESOURCES / "mvtec_ad",
    "visa": RESOURCES / "VisA",
}

ORDERING = "easy_to_hard"
STEP_SCHEDULE = "3x9"

if __name__ == "__main__":
    """
    This example combines the two features: a multidataset stream and step scheduling.

    InCLAD-MD contributes 27 categories from five source datasets. The schedule "3x9" merges
    them into nine training steps of three categories each, while evaluation stays per
    category -- so the metric matrix is 9 x 27 instead of 27 x 27.

    Because consecutive concepts are merged, a training step can straddle a source-dataset
    boundary: with "3x9" and the easy_to_hard ordering, step 1 holds the last BTech category
    together with the first two MPDD ones. Schedules that align with the block sizes
    (3 for BTech, 6 for the rest) keep steps within a single source.

    Other schedules over 27 categories: "26-1", "9x3", "21-1x6", "15-6-6".
    """
    dataset = load_inclad_md(
        roots=ROOTS,
        ordering=ORDERING,
        resize_to=(256, 256),
        max_train_samples_per_category=100,
    )

    scheduled_dataset = apply_step_schedule(dataset, schedule=STEP_SCHEDULE)

    print("Training steps:", [(c.name, c.data.shape[0]) for c in scheduled_dataset.train_concepts()])
    print("Evaluated categories:", len(scheduled_dataset.test_concepts()))
    for step_index, step_name in enumerate(c.name for c in scheduled_dataset.train_concepts()):
        members = [name for name, step in scheduled_dataset.first_seen_step().items() if step == step_index]
        sources = sorted({split_multi_dataset_concept_name(name)[0] for name in members})
        print(f"  {step_name}: {members} (sources: {sources})")

    model = RD4AD(
        RD4ADConfig(
            input_size=(256, 256),
            backbone_name="resnet18",
            pretrained_encoder=True,
            freeze_encoder=True,
            batch_size=16,
            epochs=2,
            learning_rate=5e-3,
            score_smoothing_sigma=4.0,
            score_mode="max",
            threshold_quantile=0.99,
            show_training_progress=True,
        )
    )
    strategy = NaiveStrategy(model)
    summarized_metrics = [FinalStepAverage()]
    schedule_aware_metrics = [
        ScheduleAwareForgettingMeasure(),
        ScheduleAwareForwardTransfer(),
        ScheduleAwareNewTaskAcquisition(),
    ]
    first_seen_step = scheduled_dataset.first_seen_step()

    callbacks = [
        # Image-level
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        ConceptMetricCallback(
            base_metric=AveragePrecision(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        # Pixel-level
        VisionPixelConceptMetricCallback(
            base_metric=PixelRocAuc(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        VisionPixelConceptMetricCallback(
            base_metric=PixelAUPRO(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptIncrementalScenario(dataset=scheduled_dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path(f"output_inclad_md_{ORDERING}_{STEP_SCHEDULE}.json"))
    output_writer.write([model, scheduled_dataset, strategy, *callbacks])
