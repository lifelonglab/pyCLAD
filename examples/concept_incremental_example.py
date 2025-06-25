import logging
import pathlib

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.energy_evaluation import EnergyEvaluationCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

sns.set_theme(style="darkgrid")
logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])


def _generate_normal_dist(mean, cov):
    train_data = np.random.multivariate_normal(mean, cov, (100,))
    test_data = np.concatenate(
        [
            np.random.multivariate_normal(mean, cov, (50,)),
            np.random.multivariate_normal([3 * m for m in mean], cov, (50,)),
        ]
    )
    test_labels = np.array([0] * 50 + [1] * 50)
    return train_data, test_data, test_labels


if __name__ == "__main__":
    """
    This example show how to create a simple dataset with 4 concepts and carry out a concept aware scenario with
    CumulativeStrategy and IsolationForest model.
    """
    concept1_train_data, concept1_test_data, concept1_test_labels = _generate_normal_dist((2, 2), [[1, 0], [0, 1]])
    concept2_train_data, concept2_test_data, concept2_test_labels = _generate_normal_dist((50, 50), [[1, 0], [0, 1]])
    concept3_train_data, concept3_test_data, concept3_test_labels = _generate_normal_dist((5, 5), [[1, 0], [0, 1]])
    concept4_train_data, concept4_test_data, concept4_test_labels = _generate_normal_dist((20, 20), [[1, 0], [0, 1]])

    concept1_train = Concept("concept1", data=concept1_train_data)
    concept1_test = Concept("concept1", data=concept1_test_data, labels=concept1_test_labels)

    concept2_train = Concept("concept2", data=concept2_train_data)
    concept2_test = Concept("concept2", data=concept2_test_data, labels=concept2_test_labels)

    concept3_train = Concept("concept3", data=concept3_train_data)
    concept3_test = Concept("concept3", data=concept3_test_data, labels=concept3_test_labels)

    concept4_train = Concept("concept4", data=concept4_train_data)
    concept4_test = Concept("concept4", data=concept4_test_data, labels=concept4_test_labels)

    # Build a dataset based on the previously created concepts
    dataset = ConceptsDataset(
        name="GeneratedDataset",
        train_concepts=[concept1_train, concept2_train, concept3_train, concept4_train],
        test_concepts=[concept1_test, concept2_test, concept3_test, concept4_test],
    )

    # Define model, strategy, and callbacks
    model = IsolationForestAdapter()
    strategy = CumulativeStrategy(model)
    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
        EnergyEvaluationCallback()
    ]

    # Execute the concept incremental scenario
    scenario = ConceptIncrementalScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    # Save the results
    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([model, dataset, strategy, *callbacks])

    # Plot all concepts for simple visualization of the scenario
    sns.scatterplot(x=concept1_train.data[:, 0], y=concept1_train.data[:, 1], label="concept1")
    sns.scatterplot(x=concept1_test.data[:, 0], y=concept1_test.data[:, 1], label="concept1_test")
    sns.scatterplot(x=concept2_train.data[:, 0], y=concept2_train.data[:, 1], label="concept2")
    sns.scatterplot(x=concept2_test.data[:, 0], y=concept2_test.data[:, 1], label="concept2_test")
    sns.scatterplot(x=concept3_train.data[:, 0], y=concept3_train.data[:, 1], label="concept3")
    sns.scatterplot(x=concept3_test.data[:, 0], y=concept3_test.data[:, 1], label="concept3_test")
    sns.scatterplot(x=concept4_train.data[:, 0], y=concept4_train.data[:, 1], label="concept4")
    sns.scatterplot(x=concept4_test.data[:, 0], y=concept4_test.data[:, 1], label="concept4_test")
    plt.legend()
    plt.show()
