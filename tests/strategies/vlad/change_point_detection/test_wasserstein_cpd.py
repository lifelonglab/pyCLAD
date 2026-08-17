import numpy as np
import pytest

from pyclad.strategies.vlad.change_point_detection.wasserstein_cpd import (
    WassersteinChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory

MINI_BATCH_SIZE = 4
MIN_DISTRIBUTION_SIZE = 20


def _memory() -> VladMemory:
    return VladMemory(
        memory_bound=1000,
        summarization_trigger_ratio=100.0,  # never triggers within these tests
        subconcept_threshold_ratio=1.0,
        mini_batch_size=MINI_BATCH_SIZE,
        threshold_ratio=3.0,
        min_distribution_size=MIN_DISTRIBUTION_SIZE,
        n_centroids=2,
        rng=np.random.default_rng(0),
    )


def _cluster(rng, mean, n=40, n_features=3, std=0.3):
    return rng.normal(mean, std, (n, n_features))


class TestBootstrapAndSteadyState:
    def test_first_batch_emits_a_single_new_concept_change_point(self):
        memory = _memory()
        detector = WassersteinChangePointDetector(memory)
        rng = np.random.default_rng(1)

        change_points = detector.detect(_cluster(rng, mean=0.0))

        assert len(change_points) == 1
        assert change_points[0].is_new_concept
        assert change_points[0].matched_concept_id is None
        assert memory.n_concepts() == 1

    def test_in_distribution_data_does_not_trigger_further_change_points(self):
        memory = _memory()
        detector = WassersteinChangePointDetector(memory)
        rng = np.random.default_rng(2)

        detector.detect(_cluster(rng, mean=0.0))
        change_points = detector.detect(_cluster(rng, mean=0.0))

        assert change_points == []
        assert memory.n_concepts() == 1

    def test_trailing_partial_mini_batch_is_dropped(self):
        memory = _memory()
        detector = WassersteinChangePointDetector(memory)
        rng = np.random.default_rng(3)

        change_points = detector.detect(_cluster(rng, mean=0.0, n=MINI_BATCH_SIZE + 1))

        assert memory.total_samples() == MINI_BATCH_SIZE
        assert len(change_points) == 1


class TestNewVsRecurringConcepts:
    def test_a_far_batch_spawns_a_new_concept(self):
        memory = _memory()
        detector = WassersteinChangePointDetector(memory)
        rng = np.random.default_rng(4)

        detector.detect(_cluster(rng, mean=0.0))
        change_points = detector.detect(_cluster(rng, mean=50.0))

        assert len(change_points) == 1
        assert change_points[0].is_new_concept
        assert memory.n_concepts() == 2

    def test_recurring_concept_reuses_existing_memory_node(self):
        """The crux of Lifelong Change Point Detection (paper Fig. 5): a stream A, B, A should
        end with 2 memory concepts, not 3 — the second A segment must be recognized as a
        recurrence of the first, not spawn a duplicate."""
        memory = _memory()
        detector = WassersteinChangePointDetector(memory)
        rng = np.random.default_rng(5)

        detector.detect(_cluster(rng, mean=0.0))
        concept_a_id = next(iter(memory.concept_ids()))

        detector.detect(_cluster(rng, mean=50.0))
        cps_a_again = detector.detect(_cluster(rng, mean=0.0))

        assert memory.n_concepts() == 2
        assert len(cps_a_again) == 1
        assert not cps_a_again[0].is_new_concept
        assert cps_a_again[0].matched_concept_id == concept_a_id


if __name__ == "__main__":
    pytest.main([__file__])
