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
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory
from pyclad.strategies.vlad.vlad_strategy import VladConceptIncrementalStrategy

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    VLAD in the concept-incremental scenario: task/concept boundaries are given by the scenario
    driver (one learn() call per concept), so BoundaryChangePointDetector skips online drift
    scanning, but still classifies each concept as new vs. recurring against memory.
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

    # for_concept_incremental() defaults to SlicedWassersteinDistance, validated as reliably
    # matching ExactWassersteinDistance's concept count for this mode's whole-batch comparisons
    # while being far faster (~1.7s vs. ~110s on this dataset). Pass distance=ExactWassersteinDistance()
    # explicitly if you'd rather have the paper-faithful default.
    memory = VladMemory.for_concept_incremental(
        memory_bound=1000,
        summarization_trigger_ratio=5.0,
        subconcept_threshold_ratio=1.25,
        mini_batch_size=4,
        threshold_ratio=1.25,
        min_distribution_size=256,
        n_centroids=5,
        max_comparison_size=256,
    )
    strategy = VladConceptIncrementalStrategy(model=model, memory=memory, max_steps_between_updates=5000)

    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
    ]
    scenario = ConceptIncrementalScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
