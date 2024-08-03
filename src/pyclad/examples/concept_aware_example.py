import pathlib

from pyclad.callbacks.evaluation.matrix_evaluation import MatrixMetricEvaluationCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.classical.isolation_forest import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware_scenario import concept_aware_scenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

if __name__ == "__main__":
    dataset = read_dataset_from_npy(
        pathlib.Path("resources/nsl-kdd_random_anomalies_5_concepts_1000_per_cluster.npy"), dataset_name="NSL-KDD-R"
    )
    strategy = CumulativeStrategy(IsolationForestAdapter())
    callbacks = [
        MatrixMetricEvaluationCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
    ]
    concept_aware_scenario(dataset, strategy=strategy, callbacks=callbacks)

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([dataset, strategy, *callbacks])
