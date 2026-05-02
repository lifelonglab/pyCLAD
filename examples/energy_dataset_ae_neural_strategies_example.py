import logging
import pathlib
import torch.nn as nn

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.architectural import PNNStrategy
from pyclad.strategies.regularization.lwf import LwFStrategy
from pyclad.strategies.regularization.ewc import EWCStrategy
from pyclad.strategies.replay.agem import AGEMStrategy

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])


def build_model(input_features: int = 14) -> Autoencoder:
    encoder = nn.Sequential(
        nn.Linear(input_features, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, input_features),
        nn.Sigmoid(),
    )
    return Autoencoder(encoder, decoder)


def build_callbacks():
    return [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]


def build_model_strategy(strategy_cls):
    model = build_model()
    return model, strategy_cls(model)


if __name__ == "__main__":
    """
    This example showcase how to run a concept aware scenario using the Energy dataset adopted to continual anomaly
    detection using the method proposed here <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios>
    """
    dataset = EnergyPlantsDataset(dataset_type="random_anomalies")

    strategy_builders = [
        lambda: build_model_strategy(LwFStrategy),
        lambda: build_model_strategy(EWCStrategy),
        lambda: build_model_strategy(AGEMStrategy),
        lambda: (None, PNNStrategy(build_model)),
    ]

    for strategy_builder in strategy_builders:
        model, strategy = strategy_builder()
        callbacks = build_callbacks()
        scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
        scenario.run()

        output_writer = JsonOutputWriter(pathlib.Path(f"output-energy-{strategy.name()}.json"))
        providers = [dataset, strategy, *callbacks] if model is None else [model, dataset, strategy, *callbacks]
        output_writer.write(providers)
