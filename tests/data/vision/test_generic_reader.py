"""Tests for GenericFolderReader and auto layout detection."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyclad.data.vision.generic_reader import (
    DEFAULT_NORMAL_LABELS,
    FolderLayout,
    GenericFolderReader,
)


def _write_rgb_image(path: Path, rgb: tuple[int, int, int] = (128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (4, 4), rgb)
    img.save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (4, 4), value)
    img.save(path)


# ---------------------------------------------------------------
# Layout: category/split/label  (MVTec-like)
# ---------------------------------------------------------------

class TestCategorySplitLabel:

    def _build_tree(self, root: Path) -> None:
        _write_rgb_image(root / "bottle" / "train" / "good" / "001.png")
        _write_rgb_image(root / "bottle" / "train" / "good" / "002.png")
        _write_rgb_image(root / "bottle" / "test" / "good" / "010.png")
        _write_rgb_image(root / "bottle" / "test" / "crack" / "011.png")
        _write_mask(root / "bottle" / "ground_truth" / "crack" / "011_mask.png")
        _write_rgb_image(root / "cable" / "train" / "good" / "001.png")
        _write_rgb_image(root / "cable" / "test" / "good" / "010.png")

    def test_available_categories(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout=FolderLayout.CATEGORY_SPLIT_LABEL)
        assert reader.available_categories() == ["bottle", "cable"]

    def test_index_samples(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout="category/split/label")
        samples = reader.index_samples()
        assert len(samples) == 6

        train = [s for s in samples if s.split == "train"]
        test = [s for s in samples if s.split == "test"]
        assert len(train) == 3
        assert len(test) == 3

        anomalous = [s for s in test if s.image_label == 1]
        assert len(anomalous) == 1
        assert anomalous[0].defect_type == "crack"
        assert anomalous[0].mask_path is not None

    def test_max_samples(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout=FolderLayout.CATEGORY_SPLIT_LABEL)
        samples = reader.index_samples(max_train_samples_per_category=1)
        train = [s for s in samples if s.split == "train"]
        assert len(train) == 2  # 1 per category, 2 categories

    def test_filter_categories(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout=FolderLayout.CATEGORY_SPLIT_LABEL)
        samples = reader.index_samples(categories=["bottle"])
        categories = {s.category for s in samples}
        assert categories == {"bottle"}

    def test_read_dataset(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, name="mydata", layout=FolderLayout.CATEGORY_SPLIT_LABEL)
        ds = reader.read_dataset(data_mode="paths")
        assert len(ds.train_concepts()) == 2
        assert ds.train_concepts()[0].name == "bottle"

    def test_detect_layout(self, tmp_path: Path):
        self._build_tree(tmp_path)
        assert GenericFolderReader.detect_layout(tmp_path) == FolderLayout.CATEGORY_SPLIT_LABEL


# ---------------------------------------------------------------
# Layout: split/category/label
# ---------------------------------------------------------------

class TestSplitCategoryLabel:

    def _build_tree(self, root: Path) -> None:
        _write_rgb_image(root / "train" / "bolt" / "good" / "001.png")
        _write_rgb_image(root / "test" / "bolt" / "good" / "010.png")
        _write_rgb_image(root / "test" / "bolt" / "rust" / "011.png")
        _write_rgb_image(root / "train" / "nut" / "good" / "001.png")
        _write_rgb_image(root / "test" / "nut" / "normal" / "010.png")

    def test_available_categories(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout=FolderLayout.SPLIT_CATEGORY_LABEL)
        assert reader.available_categories() == ["bolt", "nut"]

    def test_index_samples(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout="split/category/label")
        samples = reader.index_samples()

        train = [s for s in samples if s.split == "train"]
        test = [s for s in samples if s.split == "test"]
        assert len(train) == 2
        assert len(test) == 3

        anomalous = [s for s in test if s.image_label == 1]
        assert len(anomalous) == 1
        assert anomalous[0].defect_type == "rust"

        # "normal" label should also be recognized as label=0
        normal_nut = [s for s in test if s.category == "nut"]
        assert len(normal_nut) == 1
        assert normal_nut[0].image_label == 0

    def test_detect_layout(self, tmp_path: Path):
        self._build_tree(tmp_path)
        assert GenericFolderReader.detect_layout(tmp_path) == FolderLayout.SPLIT_CATEGORY_LABEL


# ---------------------------------------------------------------
# Layout: category/split  (flat images, no label subdirs)
# ---------------------------------------------------------------

class TestCategorySplit:

    def _build_tree(self, root: Path) -> None:
        _write_rgb_image(root / "widget_a" / "train" / "001.png")
        _write_rgb_image(root / "widget_a" / "train" / "002.png")
        _write_rgb_image(root / "widget_a" / "test" / "010.png")
        _write_rgb_image(root / "widget_b" / "train" / "001.png")
        _write_rgb_image(root / "widget_b" / "test" / "010.png")

    def test_available_categories(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout=FolderLayout.CATEGORY_SPLIT)
        assert reader.available_categories() == ["widget_a", "widget_b"]

    def test_index_samples_all_test_anomalous(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout="category/split")
        samples = reader.index_samples()

        train = [s for s in samples if s.split == "train"]
        test = [s for s in samples if s.split == "test"]
        assert all(s.image_label == 0 for s in train)
        assert all(s.image_label == 1 for s in test)

    def test_detect_layout(self, tmp_path: Path):
        self._build_tree(tmp_path)
        assert GenericFolderReader.detect_layout(tmp_path) == FolderLayout.CATEGORY_SPLIT


# ---------------------------------------------------------------
# Layout: split/category  (flat images, no label subdirs)
# ---------------------------------------------------------------

class TestSplitCategory:

    def _build_tree(self, root: Path) -> None:
        _write_rgb_image(root / "train" / "classA" / "001.png")
        _write_rgb_image(root / "train" / "classB" / "001.png")
        _write_rgb_image(root / "test" / "classA" / "010.png")
        _write_rgb_image(root / "test" / "classB" / "010.png")

    def test_available_categories(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout=FolderLayout.SPLIT_CATEGORY)
        assert reader.available_categories() == ["classA", "classB"]

    def test_index_samples_all_test_anomalous(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout="split/category")
        samples = reader.index_samples()

        train = [s for s in samples if s.split == "train"]
        test = [s for s in samples if s.split == "test"]
        assert len(train) == 2
        assert len(test) == 2
        assert all(s.image_label == 0 for s in train)
        assert all(s.image_label == 1 for s in test)

    def test_detect_layout(self, tmp_path: Path):
        self._build_tree(tmp_path)
        assert GenericFolderReader.detect_layout(tmp_path) == FolderLayout.SPLIT_CATEGORY


# ---------------------------------------------------------------
# Layout: split/label  (single category)
# ---------------------------------------------------------------

class TestSplitLabel:

    def _build_tree(self, root: Path) -> None:
        _write_rgb_image(root / "train" / "good" / "001.png")
        _write_rgb_image(root / "train" / "good" / "002.png")
        _write_rgb_image(root / "test" / "good" / "010.png")
        _write_rgb_image(root / "test" / "scratch" / "011.png")
        _write_rgb_image(root / "test" / "dent" / "012.png")

    def test_available_categories_single(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(
            root=tmp_path, layout=FolderLayout.SPLIT_LABEL,
            single_category_name="my_product",
        )
        assert reader.available_categories() == ["my_product"]

    def test_index_samples(self, tmp_path: Path):
        self._build_tree(tmp_path)
        reader = GenericFolderReader(root=tmp_path, layout="split/label")
        samples = reader.index_samples()

        train = [s for s in samples if s.split == "train"]
        test = [s for s in samples if s.split == "test"]
        assert len(train) == 2
        assert len(test) == 3
        assert all(s.image_label == 0 for s in train)
        assert all(s.category == "default" for s in samples)

        anomalous = [s for s in test if s.image_label == 1]
        assert len(anomalous) == 2
        assert {s.defect_type for s in anomalous} == {"scratch", "dent"}

    def test_detect_layout(self, tmp_path: Path):
        self._build_tree(tmp_path)
        assert GenericFolderReader.detect_layout(tmp_path) == FolderLayout.SPLIT_LABEL


# ---------------------------------------------------------------
# Custom normal labels
# ---------------------------------------------------------------

def test_custom_normal_labels(tmp_path: Path):
    _write_rgb_image(tmp_path / "train" / "OK" / "001.png")
    _write_rgb_image(tmp_path / "test" / "OK" / "010.png")
    _write_rgb_image(tmp_path / "test" / "defect" / "011.png")

    reader = GenericFolderReader(
        root=tmp_path,
        layout=FolderLayout.SPLIT_LABEL,
        normal_labels=frozenset({"ok"}),
    )
    samples = reader.index_samples()
    normal_test = [s for s in samples if s.split == "test" and s.image_label == 0]
    assert len(normal_test) == 1


# ---------------------------------------------------------------
# Auto-detection error case
# ---------------------------------------------------------------

def test_detect_layout_empty_dir_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no subdirectories"):
        GenericFolderReader.detect_layout(tmp_path)
