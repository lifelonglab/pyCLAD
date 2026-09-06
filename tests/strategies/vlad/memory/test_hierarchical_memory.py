import numpy as np
import pytest

from pyclad.strategies.vlad.distance import (
    ExactWassersteinDistance,
    SlicedWassersteinDistance,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory
from pyclad.strategies.vlad.memory.summarization import kmeans_summarize, pyramid_budget


def _memory(**overrides) -> VladMemory:
    params = dict(
        memory_bound=1000,
        summarization_trigger_ratio=5.0,
        subconcept_threshold_ratio=5.0,
        mini_batch_size=4,
        threshold_ratio=2.0,
        min_distribution_size=20,
        n_centroids=2,
        rng=np.random.default_rng(0),
    )
    params.update(overrides)
    return VladMemory(**params)


def _gaussian(rng, mean, n=20, n_features=3, std=0.5):
    return rng.normal(mean, std, (n, n_features))


class TestConceptCreationAndCalibration:
    def test_first_concept_is_a_root(self):
        memory = _memory()
        rng = np.random.default_rng(1)
        concept_id = memory.create_concept(_gaussian(rng, mean=0.0))

        assert memory.n_concepts() == 1
        assert memory._nodes[concept_id].parent_id is None
        assert memory._nodes[concept_id].layer == 1

    def test_concept_is_calibrated_once_min_distribution_size_reached(self):
        memory = _memory(min_distribution_size=20)
        rng = np.random.default_rng(1)

        concept_id = memory.create_concept(_gaussian(rng, mean=0.0, n=10))
        assert memory.threshold(concept_id) is None  # not enough samples yet

        memory.add_to_concept(concept_id, _gaussian(rng, mean=0.0, n=10))
        assert memory.threshold(concept_id) is not None  # 20 samples now: calibrated
        assert memory.threshold(concept_id) >= 0

    def test_calibrated_concepts_excludes_uncalibrated_ones(self):
        memory = _memory(min_distribution_size=20)
        rng = np.random.default_rng(1)
        memory.create_concept(_gaussian(rng, mean=0.0, n=5))

        assert memory.calibrated_concepts() == {}

    def test_concept_ids_includes_uncalibrated_concepts(self):
        """Unlike `calibrated_concepts()`, `concept_ids()` and `samples()` are not gated on
        calibration — they're the way to inspect a concept that's still forming."""
        memory = _memory(min_distribution_size=20)
        rng = np.random.default_rng(1)
        concept_id = memory.create_concept(_gaussian(rng, mean=0.0, n=5))

        assert memory.concept_ids() == {concept_id}
        assert len(memory.samples(concept_id)) == 5
        assert concept_id not in memory.calibrated_concepts()


class TestConsolidation:
    def test_similar_distribution_becomes_a_subconcept(self):
        memory = _memory(subconcept_threshold_ratio=10.0)
        rng = np.random.default_rng(2)

        root_id = memory.create_concept(_gaussian(rng, mean=0.0))
        sub_id = memory.create_concept(_gaussian(rng, mean=0.0))  # same distribution, new draw

        assert memory._nodes[sub_id].parent_id == root_id
        assert memory._nodes[sub_id].layer == 2

    def test_distant_distribution_becomes_a_new_root(self):
        memory = _memory(subconcept_threshold_ratio=1.0)
        rng = np.random.default_rng(3)

        memory.create_concept(_gaussian(rng, mean=0.0))
        far_id = memory.create_concept(_gaussian(rng, mean=1000.0))

        assert memory._nodes[far_id].parent_id is None
        assert memory._nodes[far_id].layer == 1


class TestSummarization:
    def test_should_summarize_once_above_bound(self):
        memory = _memory(memory_bound=10, summarization_trigger_ratio=1.0, min_distribution_size=1000)
        rng = np.random.default_rng(4)
        memory.create_concept(_gaussian(rng, mean=0.0, n=15))

        assert memory.should_summarize()

    def test_summarize_shrinks_buffers_towards_pyramid_budget(self):
        memory = _memory(memory_bound=20, n_centroids=2, min_distribution_size=1000)
        rng = np.random.default_rng(5)
        concept_id = memory.create_concept(_gaussian(rng, mean=0.0, n=50))

        memory.summarize()

        assert len(memory.samples(concept_id)) <= 20

    def test_summarize_recalibrates_thresholds_of_calibrated_concepts(self):
        memory = _memory(memory_bound=15, min_distribution_size=20, n_centroids=2)
        rng = np.random.default_rng(6)
        concept_id = memory.create_concept(_gaussian(rng, mean=0.0, n=20))
        threshold_before = memory.threshold(concept_id)

        memory.summarize()

        assert memory.threshold(concept_id) is not None
        # buffer shrank from 20 to <=15 samples, so re-chunking into mini-batches differs
        assert len(memory.samples(concept_id)) <= 15
        assert threshold_before is not None


class TestReplayBuffer:
    def test_replay_buffer_is_union_of_all_concepts(self):
        memory = _memory(subconcept_threshold_ratio=1.0)
        rng = np.random.default_rng(7)
        memory.create_concept(_gaussian(rng, mean=0.0, n=10))
        memory.create_concept(_gaussian(rng, mean=1000.0, n=15))

        assert len(memory.replay_buffer()) == 25

    def test_replay_buffer_empty_when_no_concepts(self):
        memory = _memory()
        assert len(memory.replay_buffer()) == 0


class TestPyramidBudget:
    def test_root_gets_more_budget_than_subconcept(self):
        budgets = pyramid_budget({"root": 1, "sub": 2}, memory_bound=300)

        assert budgets["root"] > budgets["sub"]
        assert budgets["root"] + budgets["sub"] == 300

    def test_empty_hierarchy_gives_empty_budget(self):
        assert pyramid_budget({}, memory_bound=300) == {}


class TestKMeansSummarize:
    def test_returns_input_unchanged_when_already_within_budget(self):
        rng = np.random.default_rng(8)
        samples = rng.normal(size=(5, 3))

        result = kmeans_summarize(samples, budget=10, n_centroids=2, rng=rng)

        np.testing.assert_array_equal(result, samples)

    def test_shrinks_to_at_most_budget(self):
        rng = np.random.default_rng(9)
        samples = rng.normal(size=(100, 3))

        result = kmeans_summarize(samples, budget=10, n_centroids=2, rng=rng)

        assert 0 < len(result) <= 10


class TestZeroThresholdConcept:
    """A concept whose calibrated threshold is exactly 0.0 (all its internal chunks are
    identical to its own buffer — e.g. a concept made entirely of duplicate rows) must stay
    matchable by an exact hit, and must not be silently dropped from the searchable pool just
    because `0.0` is falsy — nor treated as an automatic match for anything, which dividing by
    zero would do."""

    def _zero_threshold_memory(self, **overrides) -> VladMemory:
        # min_distribution_size == 2 * mini_batch_size is the smallest allowed configuration
        # (see VladMemory's validation). With every row identical, the buffer's 2 internal
        # chunks are each trivially distance-0 from the whole buffer, giving threshold == 0.0.
        return _memory(mini_batch_size=4, min_distribution_size=8, threshold_ratio=2.0, **overrides)

    def test_zero_threshold_is_reachable(self):
        memory = self._zero_threshold_memory()
        concept_id = memory.create_concept(np.zeros((8, 3)))

        assert memory.threshold(concept_id) == 0.0
        assert concept_id in memory.calibrated_concepts()

    def test_exact_match_against_a_zero_threshold_concept_is_found(self):
        memory = self._zero_threshold_memory()
        concept_id = memory.create_concept(np.zeros((8, 3)))

        match = memory.find_best_match(np.zeros((4, 3)))

        assert match is not None
        assert match[0] == concept_id
        assert match[1] == 0.0

    def test_non_exact_candidate_does_not_match_a_zero_threshold_concept(self):
        memory = self._zero_threshold_memory()
        memory.create_concept(np.zeros((8, 3)))

        match = memory.find_best_match(np.full((4, 3), 5.0))

        assert match is None

    def test_zero_threshold_concept_can_still_gain_an_exact_subconcept(self):
        memory = self._zero_threshold_memory(subconcept_threshold_ratio=1.0)
        root_id = memory.create_concept(np.zeros((8, 3)))

        sub_id = memory.create_concept(np.zeros((8, 3)))  # exact duplicate of the root

        assert memory._nodes[sub_id].parent_id == root_id

    def test_zero_threshold_concept_does_not_falsely_absorb_a_different_concept(self):
        memory = self._zero_threshold_memory(subconcept_threshold_ratio=1.0)
        memory.create_concept(np.zeros((8, 3)))

        far_id = memory.create_concept(np.full((4, 3), 100.0))

        assert memory._nodes[far_id].parent_id is None  # became its own root, not a subconcept


class TestPluggableDistance:
    def test_defaults_to_exact_wasserstein_distance(self):
        memory = _memory()

        assert isinstance(memory._wasserstein_distance, ExactWassersteinDistance)

    def test_reports_the_active_distance_in_additional_info(self):
        memory = _memory(distance=SlicedWassersteinDistance(n_projections=13))

        assert memory.additional_info()["distance"] == {
            "name": "SlicedWassersteinDistance",
            "n_projections": 13,
        }

    def test_recurring_concept_detection_still_works_with_sliced_distance(self):
        """Same scenario as the WassersteinChangePointDetector recurrence test, but exercised
        directly through VladMemory with SlicedWassersteinDistance plugged in, to confirm the
        approximation doesn't break the new-vs-recurring decision on well-separated concepts."""
        memory = _memory(
            subconcept_threshold_ratio=1.0,
            distance=SlicedWassersteinDistance(n_projections=50, rng=np.random.default_rng(1)),
        )
        rng = np.random.default_rng(2)

        concept_a = memory.create_concept(_gaussian(rng, mean=0.0, n=30))
        memory.create_concept(_gaussian(rng, mean=50.0, n=30))
        match = memory.find_best_match(_gaussian(rng, mean=0.0, n=4))

        assert memory.n_concepts() == 2
        assert match is not None
        assert match[0] == concept_a


class TestModeAwareFactories:
    """for_concept_agnostic/for_concept_incremental exist so users don't have to know which
    distance is validated as reliable for which scenario — these tests just confirm the right
    default is applied, and that it can still be overridden."""

    def test_for_concept_agnostic_defaults_to_exact_distance(self):
        memory = VladMemory.for_concept_agnostic(memory_bound=1000, mini_batch_size=4, min_distribution_size=8)

        assert isinstance(memory._wasserstein_distance, ExactWassersteinDistance)

    def test_for_concept_incremental_defaults_to_sliced_distance(self):
        memory = VladMemory.for_concept_incremental(memory_bound=1000, mini_batch_size=4, min_distribution_size=8)

        assert isinstance(memory._wasserstein_distance, SlicedWassersteinDistance)

    def test_for_concept_agnostic_distance_override_wins(self):
        override = SlicedWassersteinDistance(n_projections=5)
        memory = VladMemory.for_concept_agnostic(
            memory_bound=1000, mini_batch_size=4, min_distribution_size=8, distance=override
        )

        assert memory._wasserstein_distance is override

    def test_for_concept_incremental_distance_override_wins(self):
        override = ExactWassersteinDistance()
        memory = VladMemory.for_concept_incremental(
            memory_bound=1000, mini_batch_size=4, min_distribution_size=8, distance=override
        )

        assert memory._wasserstein_distance is override

    def test_factories_forward_other_constructor_params(self):
        memory = VladMemory.for_concept_agnostic(
            memory_bound=1000, mini_batch_size=4, min_distribution_size=8, n_centroids=7
        )

        assert memory.n_centroids == 7


if __name__ == "__main__":
    pytest.main([__file__])
