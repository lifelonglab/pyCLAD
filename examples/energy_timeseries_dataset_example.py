import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset
from pyclad.data.timeseries import convert_dataset_to_overlapping_windows
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import LocalOutlierFactorAdapter
from pyclad.models.adapters.temporal_adapter import FlattenTimeSeriesAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example showcase how to run a concept aware scenario using the Energy dataset adopted to continual anomaly
    detection using the method proposed here <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios>
    """
    dataset = EnergyPlantsDataset(dataset_type="random_anomalies")
    dataset = convert_dataset_to_overlapping_windows(window_size=10, dataset=dataset)

    model = LocalOutlierFactorAdapter()
    strategy = CumulativeStrategy(FlattenTimeSeriesAdapter(model))
    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]
    scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output-energy.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
