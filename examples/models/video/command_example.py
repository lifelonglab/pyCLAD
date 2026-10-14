"""Continual video anomaly detection on UCF-Crime with COMMAND.

Inputs and supervision: each video is a bag of 32 ordered snippets, each with
1,024 RGB appearance features and 1,024 optical-flow motion features. Training
uses video-level normal/anomalous labels; frame annotations are for evaluation.

Concepts: the dataset supplies three successive training tasks, each containing
new anomaly classes plus an equally sized, disjoint share of normal videos:
    T1: Abuse, Arrest, Arson, Assault.
    T2: Burglary, Explosion, Fighting, RoadAccidents.
    T3: Robbery, Shooting, Shoplifting, Stealing, Vandalism.

Temporality: COMMAND models snippet order within each complete video bag.
Across tasks, the scenario learns T1 -> T2 -> T3; this is a class-based learning
schedule, not the real-world chronology of the videos. ContTrain++ retains
model/optimizer state and replays complete earlier bags to limit forgetting.

Evaluation: after each task, the same full test split is scored, including
classes not yet learned. Video scores use the maximum snippet score; frame
scores expand snippet scores over their frame ranges. ROC-AUC/AP, final-step
averages, timing, and run metadata are written to output.json.
"""

import logging
import pathlib

import torch

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.video.callbacks.video_frame_metric_callback import VideoFrameMetricCallback
from pyclad.video.data.command_ucf_crime import CommandUcfCrimeDataset
from pyclad.video.metrics.frame_average_precision import FrameAveragePrecision
from pyclad.video.metrics.frame_roc_auc import FrameRocAuc
from pyclad.video.models.command.command import CommandModel
from pyclad.video.models.command.config import (
    CommandArchitectureConfig,
    CommandConfig,
)
from pyclad.video.strategies.config import ContTrainPlusPlusConfig
from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset = CommandUcfCrimeDataset()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_config = CommandConfig(
        architecture=CommandArchitectureConfig(
            appearance_dim=1024,
            motion_dim=1024,
            hidden_dim=2048,
            memory_size=256,
        ),
        learning_rate=1e-4,
        secondary_memory_learning_rate=1e-5,
        seed=42,
        device=device,
    )
    strategy_config = ContTrainPlusPlusConfig(
        epochs=20,
        batch_size=32,
        replay_batch_size=16,
        buffer_size=1000,
        lr_step_size=10,
        lr_gamma=0.1,
        seed=42,
    )
    model = CommandModel(config=model_config)
    strategy = ContTrainPlusPlusStrategy(model, config=strategy_config)

    summarized_metrics = [FinalStepAverage()]
    callbacks = [
        # Video-level
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics),
        ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=summarized_metrics),
        # Frame-level
        VideoFrameMetricCallback(base_metric=FrameRocAuc(), summarized_metrics=summarized_metrics),
        VideoFrameMetricCallback(base_metric=FrameAveragePrecision(), summarized_metrics=summarized_metrics),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
