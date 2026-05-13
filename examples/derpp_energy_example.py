import logging
import pathlib

from torch import nn

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset
from pyclad.data.timeseries import convert_dataset_to_overlapping_windows
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.regularization.der import DerPlusPlus
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])


# does not use FlattenTimeSeriesAdapter, since it flattens the data inside
# the learn and predict methods. The DerPlusPlus strategy does inject it's
# own training routine, hence overloading `learn` does not help.
def _flatten_concept(concept: Concept) -> Concept:
    flat_data = concept.data.reshape(concept.data.shape[0], -1)
    return Concept(name=concept.name, data=flat_data, labels=concept.labels)


if __name__ == "__main__":
    window_size = 10

    dataset = EnergyPlantsDataset(dataset_type="random_anomalies")
    dataset = convert_dataset_to_overlapping_windows(window_size=window_size, dataset=dataset)

    train_concepts = [_flatten_concept(concept) for concept in dataset.train_concepts()]
    test_concepts = [_flatten_concept(concept) for concept in dataset.test_concepts()]
    dataset = ConceptsDataset(
        name=dataset.name(),
        train_concepts=train_concepts,
        test_concepts=test_concepts,
    )

    feature_dim = train_concepts[0].data.shape[1]

    encoder = nn.Sequential(
        nn.Linear(feature_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, feature_dim),
        nn.Sigmoid(),
    )

    model = Autoencoder(encoder=encoder, decoder=decoder, epochs=10)
    buffer = ReservoirBuffer(max_capacity=500, device="cpu")
    strategy = DerPlusPlus(
        model=model,
        buffer=buffer,
        alpha=0.5,
        beta=0.5,
        batch_size=1024,
        lr=1e-3,
        epochs=10,
    )

    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
            stepwise_metrics=[ForgettingMeasure()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]

    scenario = ConceptAgnosticScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output-energy-derpp.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
