from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pyclad.vision.data.benchmarks.hf_benchmark import (
    InCLADBenchDataset,
    load_inclad_bench,
)
from tests.vision._helpers import write_mask as _write_helper_mask, write_rgb_image as _write_helper_rgb


def _write_rgb_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)):
    _write_helper_rgb(path, color=color, size=(8, 8))


def _write_mask(path: Path, value: int = 255):
    _write_helper_mask(path, value=value, size=(8, 8))


_COMMON_ROW_DEFAULTS = {
    "ordering_name": "easy_to_hard",
    "ordering_master_seed": 42,
    "ordering_seed": 42,
    "source_homepage": "https://example.com",
    "source_license": "MIT",
    "mask_relpath": "",
    "image_label": 0,
    "defect_type": "",
}


def _build_fake_hf_manifest(root: Path, benchmark: str = "mvtec") -> pd.DataFrame:
    cat1 = "widget"
    cat2 = "gadget"

    _write_rgb_image(root / cat1 / "train" / "good" / "000.png")
    _write_rgb_image(root / cat1 / "train" / "good" / "001.png")
    _write_rgb_image(root / cat1 / "test" / "good" / "100.png")
    _write_rgb_image(root / cat1 / "test" / "scratch" / "101.png")
    _write_mask(root / cat1 / "ground_truth" / "scratch" / "101_mask.png")

    _write_rgb_image(root / cat2 / "train" / "good" / "000.png")
    _write_rgb_image(root / cat2 / "test" / "good" / "100.png")
    _write_rgb_image(root / cat2 / "test" / "dent" / "101.png")
    _write_mask(root / cat2 / "ground_truth" / "dent" / "101_mask.png")

    def row(index: int, **overrides) -> dict:
        return {
            "sample_id": f"{benchmark}:{index:06d}",
            "source_dataset": benchmark,
            **_COMMON_ROW_DEFAULTS,
            **overrides,
        }

    rows = [
        row(0, category=cat1, category_order=1, split="train", image_relpath=f"{cat1}/train/good/000.png"),
        row(1, category=cat1, category_order=1, split="train", image_relpath=f"{cat1}/train/good/001.png"),
        row(2, category=cat1, category_order=1, split="test", image_relpath=f"{cat1}/test/good/100.png"),
        row(
            3,
            category=cat1,
            category_order=1,
            split="test",
            image_relpath=f"{cat1}/test/scratch/101.png",
            mask_relpath=f"{cat1}/ground_truth/scratch/101_mask.png",
            image_label=1,
            defect_type="scratch",
        ),
        row(4, category=cat2, category_order=2, split="train", image_relpath=f"{cat2}/train/good/000.png"),
        row(5, category=cat2, category_order=2, split="test", image_relpath=f"{cat2}/test/good/100.png"),
        row(
            6,
            category=cat2,
            category_order=2,
            split="test",
            image_relpath=f"{cat2}/test/dent/101.png",
            mask_relpath=f"{cat2}/ground_truth/dent/101_mask.png",
            image_label=1,
            defect_type="dent",
        ),
    ]

    return pd.DataFrame(rows)


class _FakeHFDataset:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self):
        return self._df


def test_hf_benchmark_loads_from_manifest(tmp_path: Path):
    root = tmp_path / "mvtec_root"
    fake_df = _build_fake_hf_manifest(root, benchmark="mvtec")

    with patch(
        "pyclad.vision.data.benchmarks.hf_benchmark.load_dataset",
        return_value=_FakeHFDataset(fake_df),
    ):
        dataset = InCLADBenchDataset(
            benchmark="mvtec",
            ordering="easy_to_hard",
            root=root,
            resize_to=(8, 8),
        )

    assert dataset.name() == "InCLAD-mvtec-easy_to_hard"
    assert len(dataset.train_concepts()) == 2
    assert len(dataset.test_concepts()) == 2

    assert dataset.train_concepts()[0].name == "widget"
    assert dataset.train_concepts()[1].name == "gadget"
    assert dataset.train_concepts()[0].data.shape[0] == 2
    assert dataset.train_concepts()[1].data.shape[0] == 1

    assert dataset.test_concepts()[0].labels is not None
    assert 1 in dataset.test_concepts()[0].labels


def test_inclad_bench_category_filtering(tmp_path: Path):
    root = tmp_path / "mvtec_root"
    fake_df = _build_fake_hf_manifest(root, benchmark="mvtec")

    with patch(
        "pyclad.vision.data.benchmarks.hf_benchmark.load_dataset",
        return_value=_FakeHFDataset(fake_df),
    ):
        dataset = InCLADBenchDataset(
            benchmark="mvtec",
            ordering="easy_to_hard",
            root=root,
            categories=["widget"],
            resize_to=(8, 8),
        )

    assert len(dataset.train_concepts()) == 1
    assert dataset.train_concepts()[0].name == "widget"


def test_inclad_bench_paths_mode(tmp_path: Path):
    root = tmp_path / "mvtec_root"
    fake_df = _build_fake_hf_manifest(root, benchmark="mvtec")

    with patch(
        "pyclad.vision.data.benchmarks.hf_benchmark.load_dataset",
        return_value=_FakeHFDataset(fake_df),
    ):
        dataset = InCLADBenchDataset(
            benchmark="mvtec",
            ordering="easy_to_hard",
            root=root,
            data_mode="paths",
        )

    assert len(dataset.train_concepts()) == 2
    for concept in dataset.train_concepts():
        for path_str in concept.data:
            assert Path(path_str).exists()


def test_load_inclad_bench_convenience(tmp_path: Path):
    root = tmp_path / "mvtec_root"
    fake_df = _build_fake_hf_manifest(root, benchmark="mvtec")

    with patch(
        "pyclad.vision.data.benchmarks.hf_benchmark.load_dataset",
        return_value=_FakeHFDataset(fake_df),
    ):
        dataset = load_inclad_bench(
            benchmark="mvtec",
            ordering="easy_to_hard",
            root=root,
            resize_to=(8, 8),
        )

    assert len(dataset.train_concepts()) == 2


def test_inclad_bench_invalid_benchmark_raises():
    with pytest.raises(ValueError, match="Unknown InCLAD-Bench benchmark"):
        InCLADBenchDataset(benchmark="nonexistent", root="/tmp")


def test_inclad_bench_invalid_ordering_raises():
    with pytest.raises(ValueError, match="Unknown ordering"):
        InCLADBenchDataset(benchmark="mvtec", ordering="invalid", root="/tmp")
