# requires: torch, torchvision
# needs local data: the Continual-MEGA benchmark, prepared as described in "Continual-MEGA" in docs/vision.md
from pyclad.callbacks import GroupedConceptMetricCallback, ScheduleAwareConceptMetricCallback
from pyclad.metrics import (
    FinalStepAverage,
    FinalStepForgettingMeasure,
    RocAuc,
    ScheduleAwareForwardTransfer,
    ScheduleAwareNewTaskAcquisition,
)
from pyclad.vision.callbacks.grouped_vision_pixel_concept_metric_callback import GroupedVisionPixelConceptMetricCallback
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision

# isort: split
# --8<-- [start:reader]
from pyclad.vision.data.benchmarks.continual_mega import ContinualMegaBenchmarkReader

reader = ContinualMegaBenchmarkReader(
    data_root="../../resources/vision/continual_mega",
    meta_dir="../../resources/vision/continual_mega/meta_files",
    scenario=2,  # 1 | 2 | 3
    task_size=30,  # classes per task: 5 | 10 | 30
    zero_shot=True,  # also evaluate on the held-out datasets
    train_samples="all",  # "all" = benchmark protocol, "normal" = one-class models only
)
dataset = reader.read_dataset()
# --8<-- [end:reader]

# --8<-- [start:grouped]
groups = dataset.group_by_concept()  # class -> task group
summarized_metrics = [FinalStepAverage(), FinalStepForgettingMeasure()]

callbacks = [
    GroupedConceptMetricCallback(RocAuc(), groups, summarized_metrics),
    GroupedVisionPixelConceptMetricCallback(PixelAveragePrecision(), groups, summarized_metrics),
]
# --8<-- [end:grouped]

# --8<-- [start:schedule_aware]
callback = ScheduleAwareConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[FinalStepAverage()],
    schedule_aware_metrics=[ScheduleAwareForwardTransfer(), ScheduleAwareNewTaskAcquisition()],
    first_seen_step=dataset.first_seen_step(),
)
# --8<-- [end:schedule_aware]
