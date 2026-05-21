from pathlib import Path

import numpy as np
from PIL import Image

from pyclad.vision.data.benchmarks.readers import MVTecBenchmarkReader
from pyclad.vision.data.generic_reader import GenericFolderReader
from pyclad.vision.data.readers.vision_reader import (
    build_vision_reader,
    index_vision_dataset,
    read_vision_dataset,
)


def _write_rgb_image(path: Path, rgb: tuple[int, int, int] = (128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (4, 4), rgb)
    img.save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (4, 4), value)
    img.save(path)


def test_build_vision_reader_dispatches_to_generic_reader_with_auto_layout(tmp_path: Path):
    _write_rgb_image(tmp_path / "bottle" / "train" / "good" / "001.png")
    _write_rgb_image(tmp_path / "bottle" / "test" / "good" / "010.png")
    _write_rgb_image(tmp_path / "bottle" / "test" / "crack" / "011.png")
    _write_mask(tmp_path / "bottle" / "ground_truth" / "crack" / "011_mask.png")

    reader = build_vision_reader(root=tmp_path, name="mydata")

    assert isinstance(reader, GenericFolderReader)

    samples = index_vision_dataset(root=tmp_path, name="mydata")
    assert len(samples) == 3
    assert any(
        sample.mask_path == tmp_path / "bottle" / "ground_truth" / "crack" / "011_mask.png"
        for sample in samples
    )

    dataset = read_vision_dataset(root=tmp_path, name="mydata", data_mode="paths")
    assert dataset.train_concepts()[0].name == "bottle"
    assert dataset.train_concepts()[0].data.dtype == object


def test_build_vision_reader_dispatches_to_benchmark_reader(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png")
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png")
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png")
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png")

    reader = build_vision_reader(root=root, benchmark="mvtec")

    assert isinstance(reader, MVTecBenchmarkReader)

    samples = index_vision_dataset(root=root, benchmark="mvtec")
    assert samples[1].mask_path == root / "widget" / "ground_truth" / "crack" / "101_mask.png"

    dataset = read_vision_dataset(root=root, benchmark="mvtec", resize_to=(6, 6))
    assert dataset.train_concepts()[0].data.shape == (1, 6, 6, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([1, 0]))
