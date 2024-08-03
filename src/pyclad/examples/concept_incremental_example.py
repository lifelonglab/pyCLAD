import pathlib

import numpy as np
from matplotlib import pyplot as plt

from pyclad.callbacks.evaluation.matrix_evaluation import MatrixMetricEvaluationCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.classical.isolation_forest import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental_scenario import concept_incremental_scenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
import seaborn as sns

sns.set_theme(style="darkgrid")

if __name__ == "__main__":
    concept1_train = Concept("concept1", data=np.random.chisquare(2, (100, 2)))
    concept1_test = Concept("concept1",
                            data=np.concatenate([np.random.chisquare(2, (50, 2)), np.random.chisquare(20, (50, 2))]),
                            labels=np.array([0] * 50 + [1] * 50))

    concept2_train = Concept("concept2", data=np.random.chisquare(100, (100, 2)))
    concept2_test = Concept("concept2",
                            data=np.concatenate([np.random.chisquare(100,(50, 2)), np.random.chisquare(150, (50, 2))]),
                            labels=np.array([0] * 50 + [1] * 50))

    concept3_train = Concept("concept3", data=np.random.chisquare(50, (100, 2)))
    concept3_test = Concept("concept3",
                            data=np.concatenate([np.random.chisquare(50, (50, 2)), np.random.chisquare(125, (50, 2))]),
                            labels=np.array([0] * 50 + [1] * 50))

    dataset = ConceptsDataset(name="GeneratedDataset", train_concepts=[concept1_train, concept2_train, concept3_train],
                              test_concepts=[concept1_test, concept2_test, concept3_test])

    strategy = CumulativeStrategy(IsolationForestAdapter())
    callbacks = [
        MatrixMetricEvaluationCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
    ]
    concept_incremental_scenario(dataset, strategy=strategy, callbacks=callbacks)

    output_writer = JsonOutputWriter(pathlib.Path("output.json"))
    output_writer.write([dataset, strategy, *callbacks])

    # Plot all concepts for simple visualization of the scenario
    sns.scatterplot(x=concept1_train.data[:, 0], y=concept1_train.data[:, 1], label='concept1')
    sns.scatterplot(x=concept1_test.data[:, 0], y=concept1_test.data[:, 1], label='concept1_test')
    sns.scatterplot(x=concept2_train.data[:, 0], y=concept2_train.data[:, 1], label='concept2')
    sns.scatterplot(x=concept2_test.data[:, 0], y=concept2_test.data[:, 1], label='concept2_test')
    sns.scatterplot(x=concept3_train.data[:, 0], y=concept3_train.data[:, 1], label='concept3')
    sns.scatterplot(x=concept3_test.data[:, 0], y=concept3_test.data[:, 1], label='concept3_test')
    plt.legend()
    plt.show()
