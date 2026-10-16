import logging
import pathlib

import numpy as np

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forgetting_measure import ForgettingMeasure
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import OneClassSVMAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example shows how to create a simple dataset with 3 concepts and carry out a concept agnostic scenario with
    CumulativeStrategy and the One-Class SVM model, measuring memory usage. Memory is measured on its own, without
    TimeEvaluationCallback, because tracemalloc slows Python down and would inflate the measured times.
    """
    seed_info = set_seed(42)

    # Three concepts centred at 0, 5 and 10; each test concept also holds anomalies shifted away from its centre
    train_concepts, test_concepts = [], []
    for i, center in enumerate([0.0, 5.0, 10.0]):
        train_concepts.append(Concept(f"concept{i + 1}", data=np.random.normal(center, 1.0, size=(300, 10))))
        normal = np.random.normal(center, 1.0, size=(90, 10))
        anomalies = np.random.normal(center + 3.0, 1.0, size=(10, 10))
        test_concepts.append(
            Concept(f"concept{i + 1}", data=np.concatenate([normal, anomalies]), labels=np.array([0] * 90 + [1] * 10))
        )

    # Build a dataset based on the previously created concepts
    dataset = ConceptsDataset(name="GeneratedDataset", train_concepts=train_concepts, test_concepts=test_concepts)

    # Define model, strategy, and callbacks
    model = OneClassSVMAdapter()
    strategy = CumulativeStrategy(model)

    metric_callback = ConceptMetricCallback(
        base_metric=RocAuc(),
        summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        stepwise_metrics=[ForgettingMeasure()],
    )
    memory_callback = MemoryUsageCallback()

    # Execute the concept agnostic scenario: the strategy receives batches of 256 samples, with no concept boundaries
    scenario = ConceptAgnosticScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback, memory_callback])
    scenario.run()

    # Save the results
    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([seed_info, model, dataset, strategy, metric_callback, memory_callback])
