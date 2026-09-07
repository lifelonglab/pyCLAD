"""Multidatasets: one concept stream drawn from several source datasets.

Each block contributes its categories under a ``<alias>__<category>`` name, and each block
resolves its images against its own root -- there is no single shared root.
"""

from pathlib import Path

import numpy as np
import pytest

from pyclad.data.grouping import apply_step_schedule
from pyclad.vision.data.benchmarks.readers import FolderBenchmarkSpec
from pyclad.vision.data.multi_dataset import (
    INCLAD_MD_ORDERINGS,
    INCLAD_MD_SPEC,
    DatasetBlock,
    MultiDatasetSpec,
    load_inclad_md,
    read_multi_dataset,
    split_multi_dataset_concept_name,
)
from pyclad.vision.data.vision_concept import VisionConcept
from tests.vision._helpers import write_mask, write_rgb_image


def _mvtec_like_tree(root: Path, categories, with_masks: bool = True, color_seed: int = 0) -> Path:
    """Write a minimal MVTec-layout tree: train/good, test/good, test/crack (+ ground_truth)."""
    for index, category in enumerate(categories):
        shade = (color_seed + index) % 256
        write_rgb_image(root / category / "train" / "good" / "000.png", color=(shade, 20, 30))
        write_rgb_image(root / category / "test" / "good" / "100.png", color=(shade, 30, 40))
        write_rgb_image(root / category / "test" / "crack" / "101.png", color=(shade, 40, 50))
        if with_masks:
            write_mask(root / category / "ground_truth" / "crack" / "101_mask.png")
    return root


@pytest.fixture
def two_sources(tmp_path):
    """Two independent roots, deliberately sharing the category name 'bottle'."""
    first = _mvtec_like_tree(tmp_path / "alpha", ["bottle", "cable"], color_seed=0)
    second = _mvtec_like_tree(tmp_path / "beta", ["bottle", "widget"], color_seed=100)
    return first, second


class TestSpec:
    def test_category_order_follows_blocks_then_categories(self):
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root="/a", categories=["cable", "bottle"]),
                DatasetBlock(dataset="visa", root="/b", categories=["pcb4"]),
            ],
        )

        assert spec.category_order() == ["mvtec__cable", "mvtec__bottle", "visa__pcb4"]

    def test_alias_overrides_the_name_prefix(self):
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[DatasetBlock(dataset="mvtec", root="/a", categories=["bottle"], alias="factory")],
        )

        assert spec.category_order() == ["factory__bottle"]

    def test_reversing_flips_both_block_and_category_order(self):
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root="/a", categories=["a", "b"]),
                DatasetBlock(dataset="visa", root="/b", categories=["x", "y"]),
            ],
        )

        assert spec.reversed().category_order() == ["visa__y", "visa__x", "mvtec__b", "mvtec__a"]

    def test_rejecting_a_spec_without_blocks(self):
        with pytest.raises(ValueError, match="at least one block"):
            MultiDatasetSpec(name="toy", blocks=[])

    def test_rejecting_duplicate_concept_names(self):
        with pytest.raises(ValueError, match="Duplicate"):
            MultiDatasetSpec(
                name="toy",
                blocks=[
                    DatasetBlock(dataset="mvtec", root="/a", categories=["bottle"]),
                    DatasetBlock(dataset="mvtec", root="/b", categories=["bottle"]),
                ],
            )

    def test_round_trip_through_a_dict(self):
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root="/a", categories=["bottle"], alias="factory"),
                DatasetBlock(dataset="visa", root="/b"),
            ],
        )

        assert MultiDatasetSpec.from_dict(spec.to_dict()) == spec

    def test_round_trip_of_a_block_using_a_custom_benchmark_spec(self):
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[DatasetBlock(dataset=FolderBenchmarkSpec(name="inhouse", train_split_dir="TRAIN"), root="/a")],
        )

        restored = MultiDatasetSpec.from_dict(spec.to_dict())

        assert restored == spec
        assert restored.blocks[0].dataset.train_split_dir == "TRAIN"

    def test_round_trip_through_a_json_file(self, tmp_path):
        spec = MultiDatasetSpec(name="toy", blocks=[DatasetBlock(dataset="mvtec", root="/a", categories=["bottle"])])
        path = tmp_path / "spec.json"

        spec.to_json(path)

        assert MultiDatasetSpec.from_json(path) == spec


class TestReading:
    def test_preserving_block_order_and_within_block_order(self, two_sources):
        first, second = two_sources
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root=second, categories=["widget", "bottle"], alias="beta"),
                DatasetBlock(dataset="mvtec", root=first, categories=["cable", "bottle"], alias="alpha"),
            ],
        )

        dataset = read_multi_dataset(spec, resize_to=(4, 4))

        assert [c.name for c in dataset.train_concepts()] == [
            "beta__widget",
            "beta__bottle",
            "alpha__cable",
            "alpha__bottle",
        ]

    def test_prefixing_keeps_colliding_category_names_apart(self, two_sources):
        first, second = two_sources
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root=first, categories=["bottle"], alias="alpha"),
                DatasetBlock(dataset="mvtec", root=second, categories=["bottle"], alias="beta"),
            ],
        )

        dataset = read_multi_dataset(spec, resize_to=(4, 4))
        names = [c.name for c in dataset.train_concepts()]

        assert names == ["alpha__bottle", "beta__bottle"]
        # Same relative path in both sources must resolve to two different images.
        first_pixels = dataset.train_concepts()[0].data[0]
        second_pixels = dataset.train_concepts()[1].data[0]
        assert not np.array_equal(first_pixels, second_pixels)

    def test_test_concepts_carry_masks_resolved_against_their_own_root(self, two_sources):
        first, second = two_sources
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root=first, categories=["bottle"], alias="alpha"),
                DatasetBlock(dataset="mvtec", root=second, categories=["widget"], alias="beta"),
            ],
        )

        dataset = read_multi_dataset(spec, resize_to=(4, 4))

        for concept in dataset.test_concepts():
            assert isinstance(concept, VisionConcept)
            assert concept.masks is not None
            assert concept.masks.shape[0] == concept.data.shape[0]

    def test_taking_every_category_when_the_block_lists_none(self, two_sources):
        first, _ = two_sources
        spec = MultiDatasetSpec(name="toy", blocks=[DatasetBlock(dataset="mvtec", root=first, alias="alpha")])

        dataset = read_multi_dataset(spec, resize_to=(4, 4))

        assert [c.name for c in dataset.train_concepts()] == ["alpha__bottle", "alpha__cable"]

    def test_resolving_roots_from_the_roots_mapping(self, two_sources):
        first, _ = two_sources
        spec = MultiDatasetSpec(name="toy", blocks=[DatasetBlock(dataset="mvtec", categories=["bottle"])])

        dataset = read_multi_dataset(spec, roots={"mvtec": first}, resize_to=(4, 4))

        assert [c.name for c in dataset.train_concepts()] == ["mvtec__bottle"]

    def test_rejecting_a_block_without_a_resolvable_root(self):
        spec = MultiDatasetSpec(name="toy", blocks=[DatasetBlock(dataset="mvtec", categories=["bottle"])])

        with pytest.raises(ValueError, match="mvtec"):
            read_multi_dataset(spec, roots={"visa": "/b"})

    def test_rejecting_an_unknown_category(self, two_sources):
        first, _ = two_sources
        spec = MultiDatasetSpec(
            name="toy", blocks=[DatasetBlock(dataset="mvtec", root=first, categories=["nonexistent"])]
        )

        with pytest.raises(ValueError, match="nonexistent"):
            read_multi_dataset(spec, resize_to=(4, 4))

    def test_applying_sample_limits_per_category(self, two_sources):
        first, _ = two_sources
        spec = MultiDatasetSpec(name="toy", blocks=[DatasetBlock(dataset="mvtec", root=first, alias="alpha")])

        dataset = read_multi_dataset(spec, resize_to=(4, 4), max_test_samples_per_category=1)

        assert all(c.data.shape[0] == 1 for c in dataset.test_concepts())


class TestConceptNames:
    def test_splitting_a_prefixed_concept_name(self):
        assert split_multi_dataset_concept_name("mvtec__bottle") == ("mvtec", "bottle")

    def test_splitting_keeps_underscores_inside_the_category(self):
        assert split_multi_dataset_concept_name("mpdd__bracket_brown") == ("mpdd", "bracket_brown")

    def test_rejecting_an_unprefixed_name(self):
        with pytest.raises(ValueError, match="separator"):
            split_multi_dataset_concept_name("bottle")


class TestInCladMd:
    def test_spec_holds_the_published_27_categories(self):
        assert len(INCLAD_MD_SPEC.category_order()) == 27

    def test_block_order_is_easiest_first(self):
        assert [block.alias or block.dataset for block in INCLAD_MD_SPEC.blocks] == [
            "btech",
            "mpdd",
            "dagm",
            "visa",
            "mvtec",
        ]

    def test_btech_contributes_all_three_categories(self):
        btech = [name for name in INCLAD_MD_SPEC.category_order() if name.startswith("btech__")]

        assert btech == ["btech__03", "btech__01", "btech__02"]

    def test_every_other_source_contributes_six_categories(self):
        for source in ("mpdd", "dagm", "visa", "mvtec"):
            selected = [n for n in INCLAD_MD_SPEC.category_order() if n.startswith(f"{source}__")]
            assert len(selected) == 6, source

    def test_supported_orderings(self):
        assert INCLAD_MD_ORDERINGS == ("easy_to_hard", "hard_to_easy")

    def test_hard_to_easy_is_the_exact_reverse(self, two_sources):
        assert INCLAD_MD_SPEC.reversed().category_order() == list(reversed(INCLAD_MD_SPEC.category_order()))

    def test_rejecting_the_random_ordering(self):
        with pytest.raises(ValueError, match="random"):
            load_inclad_md(roots={}, ordering="random")


class TestCompositionWithStepSchedule:
    def test_grouping_a_multidataset_stream(self, two_sources):
        first, second = two_sources
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root=first, categories=["bottle", "cable"], alias="alpha"),
                DatasetBlock(dataset="mvtec", root=second, categories=["widget"], alias="beta"),
            ],
        )

        dataset = read_multi_dataset(spec, resize_to=(4, 4))
        scheduled = apply_step_schedule(dataset, "2-1")

        assert [c.name for c in scheduled.train_concepts()] == ["step_0", "step_1"]
        assert scheduled.first_seen_step() == {"alpha__bottle": 0, "alpha__cable": 0, "beta__widget": 1}


class TestCategoryFilter:
    def test_selecting_a_subset_of_the_stream(self, two_sources):
        first, second = two_sources
        spec = MultiDatasetSpec(
            name="toy",
            blocks=[
                DatasetBlock(dataset="mvtec", root=first, categories=["bottle", "cable"], alias="alpha"),
                DatasetBlock(dataset="mvtec", root=second, categories=["widget"], alias="beta"),
            ],
        )

        dataset = read_multi_dataset(spec, categories=["beta__widget", "alpha__bottle"], resize_to=(4, 4))

        assert [c.name for c in dataset.train_concepts()] == ["beta__widget", "alpha__bottle"]
