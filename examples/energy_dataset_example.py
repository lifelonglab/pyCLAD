import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import LocalOutlierFactorAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """ 
    This example showcase how to run a concept aware scenario using the Energy dataset adopted to continual anomaly 
    detection using the method proposed here <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios>

    You can download the UNSW dataset adjusted to continual anomaly detection scenarios from the following link:
        <https://www.kaggle.com/datasets/nyderx/lifelong-continual-learning-for-anomaly-detection/data>
    """
    dataset_path = pathlib.Path("energy.npy")

    dataset = read_dataset_from_npy(dataset_path, dataset_name="UNSW")
    model = LocalOutlierFactorAdapter()
    strategy = CumulativeStrategy(model)
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
