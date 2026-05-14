import logging
import pathlib

import numpy as np
from torch import nn

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario
from pyclad.strategies.regularization.der import DerPlusPlus
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    feature_dim = 10

    concept1_train = Concept("concept1", data=np.random.rand(100, feature_dim))
    concept1_test = Concept("concept1", data=np.random.rand(100, feature_dim), labels=np.random.randint(0, 2, 100))

    concept2_train = Concept("concept2", data=np.random.rand(100, feature_dim))
    concept2_test = Concept("concept2", data=np.random.rand(100, feature_dim), labels=np.random.randint(0, 2, 100))

    concept3_train = Concept("concept3", data=np.random.rand(100, feature_dim))
    concept3_test = Concept("concept3", data=np.random.rand(100, feature_dim), labels=np.random.randint(0, 2, 100))

    dataset = ConceptsDataset(
        name="GeneratedDataset",
        train_concepts=[concept1_train, concept2_train, concept3_train],
        test_concepts=[concept1_test, concept2_test, concept3_test],
    )

    encoder = nn.Sequential(
        nn.Linear(feature_dim, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, feature_dim),
        nn.Sigmoid(),
    )

    model = Autoencoder(encoder=encoder, decoder=decoder)
    buffer = ReservoirBuffer(max_capacity=200, device="cpu")
    strategy = DerPlusPlus(
        model=model,
        buffer=buffer,
        alpha=0.5,
        beta=0.5,
        batch_size=32,
    )

    time_callback = TimeEvaluationCallback()
    metric_callback = ConceptMetricCallback(
        base_metric=RocAuc(),
        summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        stepwise_metrics=[ForgettingMeasure()],
    )
    memory_callback = MemoryUsageCallback()

    scenario = ConceptAgnosticScenario(
        dataset=dataset, strategy=strategy, callbacks=[metric_callback, time_callback, memory_callback]
    )
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, metric_callback, time_callback, memory_callback])
