import logging
import pathlib

import numpy as np

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.timeseries import convert_to_overlapping_windows
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import OneClassSVMAdapter
from pyclad.models.adapters.temporal_adapter import FlattenTimeSeriesAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example show how to create a simple dataset with 3 concepts and carry out a concept agnostic scenario with
    CumulativeStrategy and OneCLassSVM model in a timeseries-oriented manner. 
    Please note that the anomaly detection results will be random (0.5), as we generate random data and random labels
    """
    window_size = 10  # Define the window size for overlapping windows

    # Prepare random data for 3 concepts
    train_concepts = []
    test_concepts = []

    for i in range(3):
        train_data, _ = convert_to_overlapping_windows(window_size=window_size, data=np.random.rand(100, 10))
        test_data, test_labels = convert_to_overlapping_windows(
            window_size=window_size, data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100)
        )

        train_concepts.append(Concept(f"concept{i + 1}", data=train_data))
        test_concepts.append(Concept(f"concept{i + 1}", data=test_data, labels=test_labels))

    # Build a dataset based on the previously created concepts
    dataset = ConceptsDataset(
        name="GeneratedDataset",
        train_concepts=train_concepts,
        test_concepts=test_concepts,
    )
    # Define model, strategy, and callbacks
    model = FlattenTimeSeriesAdapter(OneClassSVMAdapter())
    strategy = CumulativeStrategy(model)

    time_callback = TimeEvaluationCallback()
    metric_callback = ConceptMetricCallback(
        base_metric=RocAuc(), metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()]
    )
    memory_callback = MemoryUsageCallback()

    # Execute the concept agnostic scenario
    scenario = ConceptAgnosticScenario(
        dataset=dataset, strategy=strategy, callbacks=[metric_callback, time_callback, memory_callback]
    )
    scenario.run()

    # Save the results
    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, metric_callback, time_callback, memory_callback])
