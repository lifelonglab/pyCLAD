import logging
import pathlib

from torch import nn

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.autoencoder.autoencoder import VariationalAutoencoder
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example showcases how to use the variational autoencoder backbone.
    """
    dataset = read_dataset_from_npy(
        pathlib.Path("resources/nsl-kdd_random_anomalies_5_concepts_1000_per_cluster.npy"), dataset_name="NSL-KDD-R"
    )
    dataset_input_features = 41
    hidden_dim = 8
    latent_dim = 4

    encoder = nn.Sequential(
        nn.Linear(dataset_input_features, 16),
        nn.ReLU(),
        nn.Linear(16, hidden_dim),
        nn.ReLU(),
    )

    decoder = nn.Sequential(
        nn.Linear(latent_dim, 16),
        nn.ReLU(),
        nn.Linear(16, dataset_input_features),
        nn.Sigmoid(),
    )

    backbone = VariationalAutoencoder(encoder, decoder, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model = TorchModelAdapter(backbone, StandardRunner(max_epochs=20), batch_size=32)

    replay_buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=1000)
    strategy = ReplayEnhancedStrategy(model, replay_buffer)
    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
    ]
    scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
