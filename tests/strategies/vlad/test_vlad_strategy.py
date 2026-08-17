import numpy as np
import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory
from pyclad.strategies.vlad.vlad_strategy import (
    VladConceptAgnosticStrategy,
    VladConceptIncrementalStrategy,
)
from tests.strategies.smoke_tests.base import BaseStrategyTest


def _memory() -> VladMemory:
    return VladMemory(
        memory_bound=1000,
        summarization_trigger_ratio=5.0,
        subconcept_threshold_ratio=5.0,
        mini_batch_size=4,
        threshold_ratio=2.0,
        min_distribution_size=8,
        n_centroids=5,
    )


class TestVladConceptAgnosticSmoke(BaseStrategyTest):
    """VLAD in the concept-agnostic scenario: boundaries are discovered online via
    WassersteinChangePointDetector, exactly as published."""

    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        memory = _memory()
        return VladConceptAgnosticStrategy(
            model=TorchModelAdapter(backbone, StandardRunner(max_epochs=2), batch_size=16),
            memory=memory,
            max_steps_between_updates=10_000,
        )


class TestVladConceptIncrementalSmoke(BaseStrategyTest):
    """VLAD in the concept-incremental scenario: boundaries are given by the scenario driver
    (one learn() call per concept) via BoundaryChangePointDetector, which still classifies each
    concept as new vs. recurring against memory."""

    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        memory = _memory()
        return VladConceptIncrementalStrategy(
            model=TorchModelAdapter(backbone, StandardRunner(max_epochs=2), batch_size=16),
            memory=memory,
            max_steps_between_updates=10_000,
        )

    @pytest.fixture(scope="class")
    def results(self, strategy):
        rng = np.random.default_rng(42)
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
        ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback]).run()
        return metric_callback.info()["concept_metric_callback_ROC-AUC"]


if __name__ == "__main__":
    pytest.main([__file__])
