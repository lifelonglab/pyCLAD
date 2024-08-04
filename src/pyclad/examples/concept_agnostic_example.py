import pathlib

import numpy as np

from pyclad.callbacks.evaluation.matrix_evaluation import MatrixMetricEvaluationCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import OneClassSVMAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_agnostic_scenario import concept_agnostic_scenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

if __name__ == "__main__":
    """
    This example show how to create a simple dataset with 3 concepts and carry out a concept agnostic scenario with
    CumulativeStrategy and IsolationForestAdapter model.
    """

    concept1_train = Concept("concept1", data=np.random.rand(100, 10))
    concept1_test = Concept("concept1", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

    concept2_train = Concept("concept2", data=np.random.rand(100, 10))
    concept2_test = Concept("concept2", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

    concept3_train = Concept("concept3", data=np.random.rand(100, 10))
    concept3_test = Concept("concept3", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

    dataset = ConceptsDataset(
        name="GeneratedDataset",
        train_concepts=[concept1_train, concept2_train, concept3_train],
        test_concepts=[concept1_test, concept2_test, concept3_test],
    )
    model = OneClassSVMAdapter()
    strategy = CumulativeStrategy(model)
    callbacks = [
        MatrixMetricEvaluationCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
    ]
    concept_agnostic_scenario(dataset, strategy=strategy, callbacks=callbacks, batch_size=64)

    output_writer = JsonOutputWriter(pathlib.Path("output_if.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
