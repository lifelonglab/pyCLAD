import numpy as np
import pytest

from pyclad.strategies.vlad.change_point_detection.boundary_cpd import (
    BoundaryChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory


def _memory() -> VladMemory:
    return VladMemory(
        memory_bound=1000,
        summarization_trigger_ratio=100.0,
        subconcept_threshold_ratio=1.0,
        mini_batch_size=4,
        threshold_ratio=3.0,
        min_distribution_size=20,
        n_centroids=2,
        rng=np.random.default_rng(0),
    )


def _cluster(rng, mean, n=40, n_features=3, std=0.3):
    return rng.normal(mean, std, (n, n_features))


class TestBoundaryChangePointDetector:
    def test_first_call_creates_a_new_concept(self):
        memory = _memory()
        detector = BoundaryChangePointDetector(memory)
        rng = np.random.default_rng(1)

        change_points = detector.detect(_cluster(rng, mean=0.0))

        assert len(change_points) == 1
        assert change_points[0].is_new_concept
        assert memory.n_concepts() == 1
        assert len(memory.calibrated_concepts()) == 1

    def test_second_call_from_a_different_distribution_is_a_new_concept(self):
        memory = _memory()
        detector = BoundaryChangePointDetector(memory)
        rng = np.random.default_rng(2)

        detector.detect(_cluster(rng, mean=0.0))
        change_points = detector.detect(_cluster(rng, mean=50.0))

        assert change_points[0].is_new_concept
        assert memory.n_concepts() == 2

    def test_recurring_concept_is_matched_without_scanning_within_the_batch(self):
        memory = _memory()
        detector = BoundaryChangePointDetector(memory)
        rng = np.random.default_rng(3)

        detector.detect(_cluster(rng, mean=0.0))
        concept_a_id = next(iter(memory.concept_ids()))
        detector.detect(_cluster(rng, mean=50.0))
        change_points = detector.detect(_cluster(rng, mean=0.0))

        assert memory.n_concepts() == 2
        assert len(change_points) == 1
        assert not change_points[0].is_new_concept
        assert change_points[0].matched_concept_id == concept_a_id

    def test_whole_batch_is_treated_as_a_single_segment(self):
        memory = _memory()
        detector = BoundaryChangePointDetector(memory)
        rng = np.random.default_rng(4)

        change_points = detector.detect(_cluster(rng, mean=0.0, n=37))

        assert len(change_points) == 1
        assert change_points[0].index == 0
        assert len(change_points[0].segment) == 37

    def test_empty_batch_returns_no_change_points_and_does_not_touch_memory(self):
        memory = _memory()
        detector = BoundaryChangePointDetector(memory)

        change_points = detector.detect(np.empty((0, 3)))

        assert change_points == []
        assert memory.n_concepts() == 0


if __name__ == "__main__":
    pytest.main([__file__])
