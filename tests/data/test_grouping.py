import numpy as np
import pytest

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.grouping import (
    StepScheduledConceptsDataset,
    apply_step_schedule,
    compute_first_seen_step,
    first_seen_step_for_test_order,
    format_step_schedule,
    group_concepts_by_schedule,
    parse_step_schedule,
)
from pyclad.vision.data.vision_concept import VisionConcept

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def _concept(name: str, rows: int, *, with_labels: bool = False, value: float = 0.0) -> Concept:
    return Concept(
        name=name,
        data=np.full((rows, 2), value, dtype=np.float64),
        labels=np.zeros(rows, dtype=np.int64) if with_labels else None,
    )


def _vision_concept(name: str, rows: int) -> VisionConcept:
    return VisionConcept(
        name=name,
        data=np.zeros((rows, 4, 4, 3), dtype=np.uint8),
        labels=np.zeros(rows, dtype=np.int64),
        masks=np.zeros((rows, 4, 4), dtype=np.uint8),
    )


class TestParseStepSchedule:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("14-1", [14, 1]),
            ("10-5", [10, 5]),
            ("3x5", [3, 3, 3, 3, 3]),
            ("3×5", [3, 3, 3, 3, 3]),
            ("3*5", [3, 3, 3, 3, 3]),
            ("10-1x5", [10, 1, 1, 1, 1, 1]),
            ("8-1x4", [8, 1, 1, 1, 1]),
            ("15", [15]),
            (" 14 - 1 ", [14, 1]),
        ],
    )
    def test_parsing_string_specs(self, spec, expected):
        assert parse_step_schedule(spec) == expected

    def test_parsing_sequence_of_ints(self):
        assert parse_step_schedule([10, 5]) == [10, 5]

    def test_accepting_matching_total_categories(self):
        assert parse_step_schedule("14-1", total_categories=15) == [14, 1]

    def test_rejecting_schedule_that_does_not_sum_to_total_categories(self):
        with pytest.raises(ValueError, match="sums to 15"):
            parse_step_schedule("14-1", total_categories=12)

    @pytest.mark.parametrize("spec", ["", "   ", "0-5", "5-0", "a", "14-", "-14", "3x0", "1.5"])
    def test_rejecting_invalid_specs(self, spec):
        with pytest.raises(ValueError):
            parse_step_schedule(spec)

    def test_rejecting_empty_sequence(self):
        with pytest.raises(ValueError):
            parse_step_schedule([])

    def test_rejecting_non_positive_entries_in_sequence(self):
        with pytest.raises(ValueError):
            parse_step_schedule([3, 0, 2])


class TestFormatStepSchedule:
    @pytest.mark.parametrize(
        "schedule, expected",
        [
            ([14, 1], "14-1"),
            ([10, 1, 1, 1], "10-1x3"),
            ([3, 3, 3, 3, 3], "3x5"),
            ([15], "15"),
        ],
    )
    def test_formatting_is_inverse_of_parsing(self, schedule, expected):
        assert format_step_schedule(schedule) == expected
        assert parse_step_schedule(expected) == schedule


class TestGroupConceptsBySchedule:
    def test_grouping_produces_one_concept_per_step(self):
        concepts = [_concept(f"c{i}", rows=2) for i in range(5)]

        grouped = group_concepts_by_schedule(concepts, [3, 2])

        assert len(grouped) == 2
        assert [c.data.shape[0] for c in grouped] == [6, 4]

    def test_indexed_naming_is_the_default(self):
        concepts = [_concept(f"c{i}", rows=1) for i in range(3)]

        grouped = group_concepts_by_schedule(concepts, [2, 1])

        assert [c.name for c in grouped] == ["step_0", "step_1"]

    def test_joined_naming_uses_original_category_names(self):
        concepts = [_concept(name, rows=1) for name in ["bottle", "cable", "capsule"]]

        grouped = group_concepts_by_schedule(concepts, [2, 1], name_mode="joined")

        assert [c.name for c in grouped] == ["bottle+cable", "capsule"]

    def test_concatenating_data_in_schedule_order(self):
        concepts = [_concept("a", rows=1, value=1.0), _concept("b", rows=1, value=2.0)]

        grouped = group_concepts_by_schedule(concepts, [2])

        np.testing.assert_array_equal(grouped[0].data, np.array([[1.0, 1.0], [2.0, 2.0]]))

    def test_concatenating_labels_when_all_concepts_have_them(self):
        concepts = [_concept(f"c{i}", rows=2, with_labels=True) for i in range(2)]

        grouped = group_concepts_by_schedule(concepts, [2])

        assert grouped[0].labels.shape == (4,)

    def test_keeping_labels_none_when_no_concept_has_them(self):
        concepts = [_concept(f"c{i}", rows=2) for i in range(2)]

        grouped = group_concepts_by_schedule(concepts, [2])

        assert grouped[0].labels is None

    def test_rejecting_mix_of_labelled_and_unlabelled_concepts(self):
        concepts = [_concept("a", rows=2, with_labels=True), _concept("b", rows=2)]

        with pytest.raises(ValueError, match="labels"):
            group_concepts_by_schedule(concepts, [2])

    def test_rejecting_schedule_that_does_not_match_concept_count(self):
        concepts = [_concept(f"c{i}", rows=1) for i in range(5)]

        with pytest.raises(ValueError, match="does not match"):
            group_concepts_by_schedule(concepts, [3, 3])

    def test_rejecting_unknown_name_mode(self):
        concepts = [_concept("a", rows=1)]

        with pytest.raises(ValueError, match="name_mode"):
            group_concepts_by_schedule(concepts, [1], name_mode="fancy")

    def test_preserving_vision_concept_type_and_masks(self):
        concepts = [_vision_concept("a", rows=2), _vision_concept("b", rows=3)]

        grouped = group_concepts_by_schedule(concepts, [2])

        assert isinstance(grouped[0], VisionConcept)
        assert grouped[0].masks.shape == (5, 4, 4)
        assert grouped[0].data.shape[0] == grouped[0].masks.shape[0]

    def test_preserving_masks_for_single_concept_step(self):
        concepts = [_vision_concept("a", rows=2)]

        grouped = group_concepts_by_schedule(concepts, [1])

        assert isinstance(grouped[0], VisionConcept)
        assert grouped[0].masks.shape == (2, 4, 4)
        assert grouped[0].name == "step_0"

    def test_rejecting_mixed_concept_types_within_a_step(self):
        concepts = [_vision_concept("a", rows=2), _concept("b", rows=2, with_labels=True)]

        with pytest.raises(ValueError, match="type"):
            group_concepts_by_schedule(concepts, [2])


class TestComputeFirstSeenStep:
    def test_mvtec_14_1(self):
        assert compute_first_seen_step(MVTEC_CATEGORIES, "14-1") == [0] * 14 + [1]

    def test_mvtec_3x5(self):
        expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

        assert compute_first_seen_step(MVTEC_CATEGORIES, "3x5") == expected

    def test_one_category_per_step_is_the_identity(self):
        names = ["a", "b", "c"]

        assert compute_first_seen_step(names, [1, 1, 1]) == [0, 1, 2]

    def test_rejecting_schedule_that_does_not_cover_all_categories(self):
        with pytest.raises(ValueError):
            compute_first_seen_step(["a", "b", "c"], "14-1")


class TestFirstSeenStepForTestOrder:
    def test_realigning_to_a_shuffled_test_order(self):
        train_order = ["a", "b", "c"]
        test_order = ["c", "a", "b"]

        assert first_seen_step_for_test_order(train_order, [2, 1], test_order) == [1, 0, 0]

    def test_raising_for_a_test_concept_missing_from_training_order(self):
        with pytest.raises(KeyError, match="unknown"):
            first_seen_step_for_test_order(["a", "b"], [2], ["a", "unknown"])


class TestApplyStepSchedule:
    def _dataset(self, train_count: int = 5, test_count: int = 5) -> ConceptsDataset:
        return ConceptsDataset(
            name="toy",
            train_concepts=[_concept(f"c{i}", rows=2) for i in range(train_count)],
            test_concepts=[_concept(f"c{i}", rows=2, with_labels=True) for i in range(test_count)],
        )

    def test_grouping_train_concepts_while_keeping_test_concepts_per_category(self):
        scheduled = apply_step_schedule(self._dataset(), "3-2")

        assert len(scheduled.train_concepts()) == 2
        assert len(scheduled.test_concepts()) == 5

    def test_keeping_original_test_concept_objects_untouched(self):
        dataset = self._dataset()

        scheduled = apply_step_schedule(dataset, "3-2")

        assert [c.name for c in scheduled.test_concepts()] == [c.name for c in dataset.test_concepts()]

    def test_grouping_test_concepts_when_requested(self):
        scheduled = apply_step_schedule(self._dataset(), "3-2", group_test=True)

        assert len(scheduled.train_concepts()) == 2
        assert len(scheduled.test_concepts()) == 2

    def test_rejecting_group_test_when_train_and_test_do_not_align(self):
        dataset = self._dataset(train_count=5, test_count=4)

        with pytest.raises(ValueError, match="1:1"):
            apply_step_schedule(dataset, "3-2", group_test=True)

    def test_deriving_dataset_name_from_schedule(self):
        scheduled = apply_step_schedule(self._dataset(), "3-2")

        assert scheduled.name() == "toy__3-2"

    def test_honouring_explicit_dataset_name(self):
        scheduled = apply_step_schedule(self._dataset(), "3-2", dataset_name="custom")

        assert scheduled.name() == "custom"

    def test_exposing_the_schedule_and_the_pre_grouping_category_order(self):
        scheduled = apply_step_schedule(self._dataset(), "3-2")

        assert isinstance(scheduled, StepScheduledConceptsDataset)
        assert scheduled.schedule() == [3, 2]
        assert scheduled.category_order() == ["c0", "c1", "c2", "c3", "c4"]
        assert scheduled.group_test() is False

    def test_exposing_first_seen_step_keyed_by_category(self):
        scheduled = apply_step_schedule(self._dataset(), "3-2")

        assert scheduled.first_seen_step() == {"c0": 0, "c1": 0, "c2": 0, "c3": 1, "c4": 1}

    def test_reporting_schedule_metadata_in_additional_info(self):
        info = apply_step_schedule(self._dataset(), "3-2").additional_info()

        assert info["step_schedule"] == "3-2"
        assert info["step_schedule_parsed"] == [3, 2]
        assert info["train_steps"] == ["step_0", "step_1"]
        assert info["group_test"] is False
        assert info["first_seen_step"] == {"c0": 0, "c1": 0, "c2": 0, "c3": 1, "c4": 1}

    def test_rejecting_schedule_that_does_not_sum_to_train_concept_count(self):
        with pytest.raises(ValueError, match="sums to"):
            apply_step_schedule(self._dataset(), "4-2")


class TestGroupTestGuards:
    def _dataset(self, test_names):
        return ConceptsDataset(
            name="toy",
            train_concepts=[_concept(f"c{i}", rows=2) for i in range(3)],
            test_concepts=[_concept(name, rows=2, with_labels=True) for name in test_names],
        )

    def test_rejecting_group_test_when_test_concepts_are_a_different_set(self):
        with pytest.raises(ValueError, match="1:1"):
            apply_step_schedule(self._dataset(["c0", "c1", "other"]), "2-1", group_test=True)

    def test_rejecting_group_test_when_test_concepts_are_in_a_different_order(self):
        with pytest.raises(ValueError, match="1:1"):
            apply_step_schedule(self._dataset(["c2", "c0", "c1"]), "2-1", group_test=True)

    def test_first_seen_step_is_unavailable_when_test_concepts_were_grouped(self):
        scheduled = apply_step_schedule(self._dataset(["c0", "c1", "c2"]), "2-1", group_test=True)

        with pytest.raises(ValueError, match="group_test=True"):
            scheduled.first_seen_step()
