"""InCLAD benchmark: six Hugging Face configs read through one loader.

Only the Hub download is faked. Every fake config is a local CSV that goes through the real
``datasets`` CSV loader, so the column types the Hub produces -- ``"03"`` inferred as an integer,
an empty mask read as missing -- reach the loader exactly as they would in a real run.
"""

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import datasets
import pytest

from pyclad.data.grouping import apply_step_schedule
from pyclad.vision.data.benchmarks.hf_benchmark import (
    INCLAD_BENCH_CONFIGS,
    INCLAD_BENCH_HF_REPO,
    INCLAD_BENCH_ORDERINGS,
    InCLADBenchConfig,
    InCLADBenchDataset,
    InCLADBenchOrdering,
    _category_order,
    _download_manifest,
    load_inclad_bench,
    split_multi_dataset_concept_name,
)
from pyclad.vision.data.vision_concept import VisionConcept
from tests.vision._helpers import write_mask, write_rgb_image

_REAL_LOAD_DATASET = datasets.load_dataset

_SINGLE_DATASET_COLUMNS = (
    "sample_id",
    "source_dataset",
    "category",
    "category_order",
    "split",
    "image_relpath",
    "mask_relpath",
    "image_label",
    "defect_type",
    "ordering_name",
    "ordering_master_seed",
    "ordering_seed",
    "source_homepage",
    "source_license",
)
# The inclad-md files carry one more column, the category name without its source prefix.
_MULTIDATASET_COLUMNS = _SINGLE_DATASET_COLUMNS[:2] + ("source_category",) + _SINGLE_DATASET_COLUMNS[2:]


def _row(source, category, order, split, image, label=0, mask="", defect=""):
    return {
        "source_dataset": source,
        "category": category,
        "category_order": order,
        "split": split,
        "image_relpath": image,
        "mask_relpath": mask,
        "image_label": label,
        "defect_type": defect,
    }


def _write_manifest(path: Path, rows, multidataset: bool = False) -> Path:
    """Write a manifest the way the Hub stores it: empty cells for missing masks and defect types."""
    columns = _MULTIDATASET_COLUMNS if multidataset else _SINGLE_DATASET_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            extra = {"source_category": row["category"].split("__", 1)[1]} if multidataset else {}
            writer.writerow(
                {
                    "sample_id": f"sample:{index:06d}",
                    "ordering_name": "fixture",
                    "ordering_master_seed": 42,
                    "ordering_seed": 1,
                    "source_homepage": "https://example.com",
                    "source_license": "MIT",
                    **extra,
                    **row,
                }
            )
    return path


class _FakeHub:
    """Serves local CSVs per (config, split) through the real ``datasets`` CSV loader."""

    def __init__(self, cache_dir: Path):
        self._files = {}
        self._cache_dir = cache_dir
        self.calls = []

    def add(self, config: str, split: str, path: Path) -> None:
        self._files[(config, split)] = path

    def load_dataset(self, path, name=None, *, split=None, cache_dir=None, **kwargs):
        assert path == INCLAD_BENCH_HF_REPO
        self.calls.append({"config": name, "split": split, "cache_dir": cache_dir})
        data_file = self._files[(name, split)]
        return _REAL_LOAD_DATASET(
            "csv", data_files=str(data_file), split="train", cache_dir=str(self._cache_dir), **kwargs
        )

    @contextmanager
    def serving(self):
        with patch.object(datasets, "load_dataset", self.load_dataset):
            yield self


@pytest.fixture(scope="module")
def btech(tmp_path_factory):
    """A BTech-like config. Its category names have leading zeros, which the Hub's type inference
    turns into integers unless the loader reads every column as text."""
    root = tmp_path_factory.mktemp("btech")
    write_rgb_image(root / "03" / "train" / "ok" / "0000.png")
    write_rgb_image(root / "03" / "test" / "ok" / "0001.png")
    write_rgb_image(root / "03" / "test" / "ko" / "0002.png")
    write_mask(root / "03" / "ground_truth" / "ko" / "0002.png")
    write_rgb_image(root / "01" / "train" / "ok" / "0000.png")
    write_rgb_image(root / "01" / "train" / "ok" / "0001.png")
    write_rgb_image(root / "01" / "test" / "ok" / "0002.png")

    def rows(order_of_03, order_of_01):
        category_03 = [
            _row("btech", "03", order_of_03, "train", "03/train/ok/0000.png"),
            _row("btech", "03", order_of_03, "test", "03/test/ok/0001.png"),
            _row(
                "btech",
                "03",
                order_of_03,
                "test",
                "03/test/ko/0002.png",
                label=1,
                mask="03/ground_truth/ko/0002.png",
                defect="ko",
            ),
        ]
        category_01 = [
            _row("btech", "01", order_of_01, "train", "01/train/ok/0000.png"),
            _row("btech", "01", order_of_01, "train", "01/train/ok/0001.png"),
            _row("btech", "01", order_of_01, "test", "01/test/ok/0002.png"),
        ]
        return category_03 + category_01 if order_of_03 < order_of_01 else category_01 + category_03

    manifests = tmp_path_factory.mktemp("btech_manifests")
    hub = _FakeHub(tmp_path_factory.mktemp("hf_cache"))
    hub.add("btech", "easy_to_hard", _write_manifest(manifests / "easy_to_hard.csv", rows(1, 2)))
    hub.add("btech", "hard_to_easy", _write_manifest(manifests / "hard_to_easy.csv", rows(2, 1)))
    return root, hub


@pytest.fixture(scope="module")
def btech_easy_to_hard(btech):
    root, hub = btech
    with hub.serving():
        return InCLADBenchDataset(benchmark="btech", roots={"btech": root}, resize_to=(4, 4))


class TestSingleDatasetConfig:
    def test_concepts_follow_the_hf_category_order(self, btech_easy_to_hard):
        assert [c.name for c in btech_easy_to_hard.train_concepts()] == ["03", "01"]
        assert [c.name for c in btech_easy_to_hard.test_concepts()] == ["03", "01"]

    def test_train_concepts_hold_only_their_train_images(self, btech_easy_to_hard):
        assert [c.data.shape[0] for c in btech_easy_to_hard.train_concepts()] == [1, 2]

    def test_normal_test_images_without_a_mask_are_loaded(self, btech_easy_to_hard):
        anomalous_category = btech_easy_to_hard.test_concepts()[0]

        assert isinstance(anomalous_category, VisionConcept)
        assert anomalous_category.labels.tolist() == [0, 1]

    def test_hard_to_easy_reads_its_own_split(self, btech):
        root, hub = btech
        with hub.serving():
            dataset = InCLADBenchDataset(
                benchmark="btech", roots={"btech": root}, ordering="hard_to_easy", resize_to=(4, 4)
            )

        assert [c.name for c in dataset.train_concepts()] == ["01", "03"]

    def test_requested_config_split_and_cache_dir_reach_the_hub(self, btech, tmp_path):
        root, hub = btech
        with hub.serving():
            InCLADBenchDataset(
                benchmark="btech", roots={"btech": root}, ordering="hard_to_easy", cache_dir=str(tmp_path)
            )

        assert hub.calls[-1] == {"config": "btech", "split": "hard_to_easy", "cache_dir": str(tmp_path)}

    def test_selecting_a_subset_of_categories(self, btech):
        root, hub = btech
        with hub.serving():
            dataset = InCLADBenchDataset(benchmark="btech", roots={"btech": root}, categories=["01"], resize_to=(4, 4))

        assert [c.name for c in dataset.train_concepts()] == ["01"]

    def test_paths_mode_returns_files_under_the_root(self, btech):
        root, hub = btech
        with hub.serving():
            dataset = InCLADBenchDataset(benchmark="btech", roots={"btech": root}, data_mode="paths")

        paths = [Path(path) for concept in dataset.train_concepts() for path in concept.data]
        assert sorted(path.relative_to(root).as_posix() for path in paths) == [
            "01/train/ok/0000.png",
            "01/train/ok/0001.png",
            "03/train/ok/0000.png",
        ]

    def test_roots_for_other_sources_are_ignored(self, btech, tmp_path):
        root, hub = btech
        with hub.serving():
            dataset = InCLADBenchDataset(
                benchmark="btech", roots={"btech": root, "mvtec": tmp_path / "not_downloaded"}, resize_to=(4, 4)
            )

        assert [c.name for c in dataset.train_concepts()] == ["03", "01"]

    def test_convenience_function_loads_the_same_stream(self, btech):
        root, hub = btech
        with hub.serving():
            dataset = load_inclad_bench(benchmark="btech", roots={"btech": root}, resize_to=(4, 4))

        assert [c.name for c in dataset.train_concepts()] == ["03", "01"]


@pytest.fixture(scope="module")
def inclad_md(tmp_path_factory):
    """Two sources in two roots. Both hold an image at the same relative path, so a loader that
    resolved every row against a single root would read the wrong file. The stream alternates
    between the sources, so a loader that grouped concepts by source would reorder it."""
    mvtec_root = tmp_path_factory.mktemp("mvtec")
    visa_root = tmp_path_factory.mktemp("visa")
    write_rgb_image(mvtec_root / "a" / "train" / "good" / "000.png")
    write_rgb_image(mvtec_root / "a" / "test" / "good" / "100.png")
    write_rgb_image(visa_root / "a" / "train" / "good" / "000.png")
    write_rgb_image(visa_root / "a" / "test" / "good" / "100.png")
    write_rgb_image(visa_root / "b" / "train" / "good" / "000.png")
    write_rgb_image(visa_root / "b" / "test" / "good" / "100.png")
    write_rgb_image(visa_root / "b" / "test" / "bad" / "101.png")
    write_mask(visa_root / "b" / "ground_truth" / "bad" / "101_mask.png")

    rows = [
        _row("visa", "visa__a", 1, "train", "a/train/good/000.png"),
        _row("visa", "visa__a", 1, "test", "a/test/good/100.png"),
        _row("mvtec", "mvtec__a", 2, "train", "a/train/good/000.png"),
        _row("mvtec", "mvtec__a", 2, "test", "a/test/good/100.png"),
        _row("visa", "visa__b", 3, "train", "b/train/good/000.png"),
        _row("visa", "visa__b", 3, "test", "b/test/good/100.png"),
        _row(
            "visa",
            "visa__b",
            3,
            "test",
            "b/test/bad/101.png",
            label=1,
            mask="b/ground_truth/bad/101_mask.png",
            defect="bad",
        ),
    ]
    manifests = tmp_path_factory.mktemp("md_manifests")
    hub = _FakeHub(tmp_path_factory.mktemp("hf_cache"))
    hub.add("inclad-md", "easy_to_hard", _write_manifest(manifests / "easy_to_hard.csv", rows, multidataset=True))
    return {"mvtec": mvtec_root, "visa": visa_root}, hub


@pytest.fixture(scope="module")
def inclad_md_paths(inclad_md):
    roots, hub = inclad_md
    with hub.serving():
        return InCLADBenchDataset(benchmark="inclad-md", roots=roots, data_mode="paths")


class TestMultidatasetConfig:
    def test_concepts_follow_the_hf_order_across_sources(self, inclad_md_paths):
        assert [c.name for c in inclad_md_paths.train_concepts()] == ["visa__a", "mvtec__a", "visa__b"]
        assert [c.name for c in inclad_md_paths.test_concepts()] == ["visa__a", "mvtec__a", "visa__b"]

    def test_each_source_reads_images_from_its_own_root(self, inclad_md, inclad_md_paths):
        roots, _ = inclad_md
        first_image = {c.name: Path(c.data[0]) for c in inclad_md_paths.train_concepts()}

        assert first_image["mvtec__a"] == roots["mvtec"] / "a" / "train" / "good" / "000.png"
        assert first_image["visa__a"] == roots["visa"] / "a" / "train" / "good" / "000.png"

    def test_masks_are_read_from_the_source_root(self, inclad_md):
        roots, hub = inclad_md
        with hub.serving():
            dataset = InCLADBenchDataset(benchmark="inclad-md", roots=roots, resize_to=(4, 4))

        visa_b = dataset.test_concepts()[2]
        assert isinstance(visa_b, VisionConcept)
        assert visa_b.labels.tolist() == [0, 1]

    def test_selecting_categories_across_sources_keeps_the_requested_order(self, inclad_md):
        roots, hub = inclad_md
        with hub.serving():
            dataset = InCLADBenchDataset(
                benchmark="inclad-md", roots=roots, categories=["visa__b", "mvtec__a"], data_mode="paths"
            )

        assert [c.name for c in dataset.train_concepts()] == ["visa__b", "mvtec__a"]

    def test_a_step_schedule_groups_consecutive_concepts_across_sources(self, inclad_md_paths):
        scheduled = apply_step_schedule(inclad_md_paths, "2-1")

        assert scheduled.first_seen_step() == {"visa__a": 0, "mvtec__a": 0, "visa__b": 1}


class TestValidation:
    @pytest.fixture
    def hub(self, tmp_path):
        hub = _FakeHub(tmp_path / "hf_cache")
        with hub.serving():
            yield hub

    def test_unknown_config_is_rejected_before_downloading(self, hub):
        with pytest.raises(ValueError, match="Unknown InCLAD benchmark config"):
            InCLADBenchDataset(benchmark="nonexistent", roots={})

        assert hub.calls == []

    def test_unknown_ordering_is_rejected_before_downloading(self, hub):
        with pytest.raises(ValueError, match="Unknown ordering"):
            InCLADBenchDataset(benchmark="mvtec", roots={}, ordering="invalid")

        assert hub.calls == []

    def test_the_multidataset_has_no_random_ordering(self, hub):
        with pytest.raises(ValueError, match="random"):
            InCLADBenchDataset(benchmark="inclad-md", roots={}, ordering="random")

        assert hub.calls == []

    def test_a_source_without_a_root_is_named(self, inclad_md):
        roots, hub = inclad_md
        with hub.serving(), pytest.raises(ValueError, match="mvtec"):
            InCLADBenchDataset(benchmark="inclad-md", roots={"visa": roots["visa"]})

    def test_a_root_missing_on_disk_is_reported(self, btech, tmp_path):
        _, hub = btech
        with hub.serving(), pytest.raises(FileNotFoundError, match="not_downloaded"):
            InCLADBenchDataset(benchmark="btech", roots={"btech": tmp_path / "not_downloaded"})

    def test_an_unknown_requested_category_is_rejected(self, inclad_md):
        roots, hub = inclad_md
        with hub.serving(), pytest.raises(ValueError, match="visa__z"):
            InCLADBenchDataset(benchmark="inclad-md", roots=roots, categories=["visa__z"], data_mode="paths")

    def test_two_categories_sharing_one_position_are_rejected(self, btech, tmp_path):
        root, _ = btech
        hub = _FakeHub(tmp_path / "hf_cache")
        rows = [
            _row("btech", "03", 1, "train", "03/train/ok/0000.png"),
            _row("btech", "01", 1, "train", "01/train/ok/0000.png"),
        ]
        hub.add("btech", "easy_to_hard", _write_manifest(tmp_path / "tied.csv", rows))

        with hub.serving(), pytest.raises(ValueError, match="category_order"):
            InCLADBenchDataset(benchmark="btech", roots={"btech": root}, data_mode="paths")

    def test_type_hints_list_every_config_and_ordering(self):
        assert set(get_args(InCLADBenchConfig)) == set(INCLAD_BENCH_CONFIGS)
        assert set(get_args(InCLADBenchOrdering)) == {
            o for orderings in INCLAD_BENCH_ORDERINGS.values() for o in orderings
        }


class TestSplitConceptName:
    def test_splits_a_prefixed_name(self):
        assert split_multi_dataset_concept_name("mvtec__bottle") == ("mvtec", "bottle")

    def test_underscores_inside_a_category_survive(self):
        assert split_multi_dataset_concept_name("mpdd__bracket_brown") == ("mpdd", "bracket_brown")

    def test_an_unprefixed_name_is_rejected(self):
        with pytest.raises(ValueError):
            split_multi_dataset_concept_name("bottle")


@pytest.mark.longrun
class TestPublishedManifests:
    """Downloads the published manifests (no images) to check what the real Hub sends."""

    @pytest.fixture(scope="class")
    def download(self, tmp_path_factory):
        cache_dir = str(tmp_path_factory.mktemp("hf_cache"))
        return lambda config: _download_manifest(config=config, ordering="easy_to_hard", cache_dir=cache_dir)

    def test_every_cell_arrives_as_text(self, download):
        manifest = download("btech")

        assert all(isinstance(value, str) for value in manifest.to_numpy().ravel())

    def test_btech_categories_keep_their_leading_zeros(self, download):
        assert sorted(download("btech")["category"].unique()) == ["01", "02", "03"]

    def test_inclad_md_streams_27_categories_from_five_sources(self, download):
        order = _category_order(download("inclad-md"))

        assert len(order) == 27
        assert {split_multi_dataset_concept_name(name)[0] for name in order} == {
            "btech",
            "dagm",
            "mpdd",
            "mvtec",
            "visa",
        }

    def test_inclad_md_concept_prefix_names_the_source_its_images_come_from(self, download):
        manifest = download("inclad-md")
        prefixes = manifest["category"].map(lambda name: split_multi_dataset_concept_name(name)[0])

        assert (prefixes == manifest["source_dataset"]).all()
