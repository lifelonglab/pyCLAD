import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import (
    ScheduleAwareConceptMetricCallback,
)
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
    ScheduleAwareVisionPixelConceptMetricCallback,
)
from pyclad.vision.data.readers.vision_reader import read_vision_dataset
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.rd4ad.rd4ad import RD4AD

logging.basicConfig(level=logging.INFO)

STEP_SCHEDULE = "3x5"

if __name__ == "__main__":
    """
    This example showcases CDAD-style step-schedule grouping on MVTec AD.

    Instead of 15 training steps (one per category), the schedule "3x5" trains on five
    consecutive steps holding three categories each. Evaluation stays per category, so the
    metric matrix is 5 x 15 rather than 15 x 15.

    Other schedules on MVTec's 15 categories: "14-1", "10-5", "10-1x5". On VisA's 12
    categories, "8-1x4". The schedule must sum to the number of train concepts.
    """
    # 1. Read the benchmark. Concepts must already be in their final order before grouping,
    #    since grouping merges *consecutive* concepts: ordering -> grouping -> first_seen_step.
    dataset = read_vision_dataset(
        root=pathlib.Path("resources/vision/mvtec_ad"),
        benchmark="mvtec",
        resize_to=(256, 256),
        data_mode="numpy",
        color_mode="rgb",
        max_train_samples_per_category=50,
        max_test_samples_per_category=50,
    )

    # 2. Group the training stream. Test concepts stay per category (group_test=False).
    scheduled_dataset = apply_step_schedule(dataset, schedule=STEP_SCHEDULE)

    print("Training steps:", [(c.name, c.data.shape[0]) for c in scheduled_dataset.train_concepts()])
    print("Evaluated categories:", [c.name for c in scheduled_dataset.test_concepts()])
    print("First seen step:", scheduled_dataset.first_seen_step())

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
    # Any strategy works -- grouping only changes what a training step contains. Naive makes
    # the forgetting story easiest to read; swap in ReplayEnhancedStrategy to soften it.
    strategy = NaiveStrategy(model)

    # 3. Metrics. ContinualAverage, BackwardTransfer, ForwardTransfer and ForgettingMeasure
    #    walk the diagonal or the triangles of the matrix and reject a rectangular one --
    #    on a 5 x 15 matrix their formulas do not mean what they say. Use FinalStepAverage
    #    (the A-AUROC of CDAD papers) plus the schedule-aware metrics, which take the step
    #    at which each category first entered training.
    summarized_metrics = [FinalStepAverage()]
    schedule_aware_metrics = [
        ScheduleAwareForgettingMeasure(),
        ScheduleAwareForwardTransfer(),
        ScheduleAwareNewTaskAcquisition(),
    ]
    first_seen_step = scheduled_dataset.first_seen_step()

    callbacks = [
        # Image-level
        ScheduleAwareConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        ScheduleAwareConceptMetricCallback(
            base_metric=AveragePrecision(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        # Pixel-level
        ScheduleAwareVisionPixelConceptMetricCallback(
            base_metric=PixelRocAuc(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        ScheduleAwareVisionPixelConceptMetricCallback(
            base_metric=PixelAUPRO(),
            summarized_metrics=summarized_metrics,
            schedule_aware_metrics=schedule_aware_metrics,
            first_seen_step=first_seen_step,
        ),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptIncrementalScenario(dataset=scheduled_dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    # The grouped dataset reports the schedule, the train steps and first_seen_step in its
    # own info(), so the output JSON is self-describing without assembling metadata by hand.
    output_writer = JsonOutputWriter(pathlib.Path(f"output_step_schedule_mvtec_{STEP_SCHEDULE}.json"))
    output_writer.write([model, scheduled_dataset, strategy, *callbacks])
