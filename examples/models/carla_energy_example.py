import logging
import pathlib

import torch

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset
from pyclad.data.timeseries import convert_dataset_to_overlapping_windows
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.backbones.resnet import ResNet1D
from pyclad.models.contrastive.carla import Carla
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.naive import NaiveStrategy

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    dataset = EnergyPlantsDataset(dataset_type="random_anomalies")
    dataset._train_concepts = dataset._train_concepts[:3]
    dataset._test_concepts = dataset._test_concepts[:3]
    dataset = convert_dataset_to_overlapping_windows(window_size=10, dataset=dataset, step_size=1)

    num_features = dataset.train_concepts()[0].data.shape[-1]
    model = Carla(
        ResNet1D(num_features=num_features),
        projection_dim=128,
        n_classes=5,
        pretext_epochs=15,
        classification_epochs=30,
        patience=5,
        entropy_loss_weight=5.0,
        inconsistency_loss_weight=0.0,
        lr_pretext=1e-4,
        lr_classification=1e-3,
        random_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    strategy = NaiveStrategy(model)

    metric_callbacks = [
        ConceptMetricCallback(
            base_metric=base_metric,
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        )
        for base_metric in (RocAuc(), AveragePrecision(), F1Score())
    ]
    callbacks = [
        *metric_callbacks,
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]
    scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output-energy-carla.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
