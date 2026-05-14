import numpy as np
import pytest
from torch import nn

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario


class BaseStrategyTest:
    CONCEPTS = ["concept1", "concept2", "concept3"]
    FEATURE_DIM = 8
    N_SAMPLES = 60

    @pytest.fixture(scope="class")
    def backbone(self):
        encoder = nn.Sequential(nn.Linear(self.FEATURE_DIM, 4), nn.ReLU())
        decoder = nn.Sequential(nn.Linear(4, self.FEATURE_DIM), nn.Sigmoid())
        return Autoencoder(encoder=encoder, decoder=decoder)

    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def results(self, strategy):
        rng = np.random.default_rng(42)
        # Each concept has a distinct Gaussian: mean shifts by 3 and std grows slightly.
        concept_distributions = [
            (np.full(self.FEATURE_DIM, i * 3.0), np.full(self.FEATURE_DIM, 0.5 + i * 0.3))
            for i in range(len(self.CONCEPTS))
        ]
        train = [
            Concept(name, data=rng.normal(mean, std, (self.N_SAMPLES, self.FEATURE_DIM)).astype(np.float32))
            for name, (mean, std) in zip(self.CONCEPTS, concept_distributions)
        ]
        test = [
            Concept(
                name,
                data=rng.normal(mean, std, (self.N_SAMPLES, self.FEATURE_DIM)).astype(np.float32),
                labels=rng.integers(0, 2, self.N_SAMPLES),
            )
            for name, (mean, std) in zip(self.CONCEPTS, concept_distributions)
        ]
        dataset = ConceptsDataset(name="TestDataset", train_concepts=train, test_concepts=test)
        metric_callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()])
        ConceptAgnosticScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback]).run()
        return metric_callback.info()["concept_metric_callback_ROC-AUC"]

    def test_all_concepts_evaluated(self, results):
        assert results["concepts_order"] == self.CONCEPTS
        assert set(results["metric_matrix"].keys()) == set(self.CONCEPTS)

    def test_scores_are_valid(self, results):
        for concept_scores in results["metric_matrix"].values():
            for score in concept_scores.values():
                assert score is not None
                assert not np.isnan(score)

    def test_summarized_metrics_computed(self, results):
        assert ContinualAverage().name() in results["metrics"]
        assert results["metrics"][ContinualAverage().name()] is not None