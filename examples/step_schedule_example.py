import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import (
    ScheduleAwareConceptMetricCallback,
)
from pyclad.data.grouping import apply_step_schedule
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
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
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.naive import NaiveStrategy

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example showcases step-schedule grouping: instead of one training step per concept,
    consecutive concepts are merged into multi-concept training steps, while evaluation stays
    per concept. The NSL-KDD scenario below has 5 concepts and the schedule "3-2" turns them
    into 2 training steps (3 concepts, then 2), so the metric matrix becomes 2 x 5.

    Ordering matters: concepts must already be in their final order before grouping, because
    grouping merges *consecutive* concepts.
    """
    dataset = read_dataset_from_npy(
        pathlib.Path("resources/nsl-kdd_random_anomalies_5_concepts_1000_per_cluster.npy"), dataset_name="NSL-KDD-R"
    )

    scheduled_dataset = apply_step_schedule(dataset, schedule="3-2")
    print("Training steps:", [concept.name for concept in scheduled_dataset.train_concepts()])
    print("Evaluated concepts:", [concept.name for concept in scheduled_dataset.test_concepts()])
    print("First seen step:", scheduled_dataset.first_seen_step())

    model = IsolationForestAdapter()
    strategy = NaiveStrategy(model)

    # Metrics that walk the diagonal (ContinualAverage, BackwardTransfer, ForwardTransfer,
    # ForgettingMeasure) reject a rectangular matrix -- pass the schedule-aware ones instead.
    callbacks = [
        ScheduleAwareConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[FinalStepAverage()],
            schedule_aware_metrics=[
                ScheduleAwareForgettingMeasure(),
                ScheduleAwareForwardTransfer(),
                ScheduleAwareNewTaskAcquisition(),
            ],
            first_seen_step=scheduled_dataset.first_seen_step(),
        )
    ]

    scenario = ConceptAwareScenario(scheduled_dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output_step_schedule.json"))
    output_writer.write([model, scheduled_dataset, strategy, *callbacks])
