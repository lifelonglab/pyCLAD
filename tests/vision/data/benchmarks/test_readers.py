import csv
from pathlib import Path

import numpy as np

from pyclad.vision.data.benchmarks.readers import (
    BTechBenchmarkReader,
    CsvBenchmarkSpec,
    DAGMBenchmarkReader,
    FolderBenchmarkSpec,
    MVTecBenchmarkReader,
    MPDDBenchmarkReader,
    VisABenchmarkReader,
    available_vision_benchmarks,
    build_vision_benchmark_reader,
    index_vision_benchmark,
    read_vision_benchmark_dataset,
)
from tests.vision._helpers import write_mask as _write_helper_mask, write_rgb_image as _write_helper_rgb


def _write_rgb_image(path: Path, color: tuple[int, int, int]):
    _write_helper_rgb(path, color=color, size=(6, 5))


def _write_mask(path: Path, value: int = 255):
    _write_helper_mask(path, value=value, size=(6, 5))


def test_read_vision_benchmark_dataset_supports_mvtec_preset(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png")

    dataset = read_vision_benchmark_dataset(root=root, benchmark="mvtec", resize_to=(8, 8))

    assert available_vision_benchmarks() == [
        "btech",
        "dagm",
        "mpdd",
        "mvtec",
        "visa",
    ]
    assert len(dataset.train_concepts()) == 1
    assert len(dataset.test_concepts()) == 1
    assert dataset.train_concepts()[0].data.shape == (1, 8, 8, 3)
    assert dataset.test_concepts()[0].data.shape == (2, 8, 8, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([1, 0]))

    samples = index_vision_benchmark(root=root, benchmark=FolderBenchmarkSpec(name="mvtec"), categories=["widget"])
    assert samples[1].mask_path == root / "widget" / "ground_truth" / "crack" / "101_mask.png"


def test_read_vision_benchmark_dataset_supports_visa_preset(tmp_path: Path):
    root = tmp_path / "visa_like"
    _write_rgb_image(root / "candle" / "Data" / "Images" / "Normal" / "000.JPG", (10, 20, 30))
    _write_rgb_image(root / "candle" / "Data" / "Images" / "Normal" / "001.JPG", (15, 25, 35))
    _write_rgb_image(root / "candle" / "Data" / "Images" / "Anomaly" / "100.JPG", (50, 60, 70))
    _write_mask(root / "candle" / "Data" / "Masks" / "Anomaly" / "100.JPG", value=120)
    (root / "split_csv").mkdir(parents=True, exist_ok=True)
    with (root / "split_csv" / "1cls.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["object", "split", "label", "image", "mask"])
        writer.writeheader()
        writer.writerow(
            {
                "object": "candle",
                "split": "train",
                "label": "normal",
                "image": "candle/Data/Images/Normal/000.JPG",
                "mask": "",
            }
        )
        writer.writerow(
            {
                "object": "candle",
                "split": "test",
                "label": "normal",
                "image": "candle/Data/Images/Normal/001.JPG",
                "mask": "",
            }
        )
        writer.writerow(
            {
                "object": "candle",
                "split": "test",
                "label": "anomaly",
                "image": "candle/Data/Images/Anomaly/100.JPG",
                "mask": "candle/Data/Masks/Anomaly/100.JPG",
            }
        )

    dataset = read_vision_benchmark_dataset(root=root, benchmark="visa", resize_to=(4, 4))

    assert dataset.train_concepts()[0].data.shape == (1, 4, 4, 3)
    assert dataset.test_concepts()[0].data.shape == (2, 4, 4, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([0, 1]))

    samples = index_vision_benchmark(root=root, benchmark=CsvBenchmarkSpec(name="visa", csv_path="split_csv/1cls.csv"))
    assert samples[-1].defect_type == "anomaly"


def test_read_vision_benchmark_dataset_supports_custom_folder_spec_and_paths_mode(tmp_path: Path):
    root = tmp_path / "custom_like"
    _write_rgb_image(root / "fabric" / "train" / "normal" / "000.png", (10, 10, 10))
    _write_rgb_image(root / "fabric" / "eval" / "normal" / "001.png", (20, 20, 20))
    _write_rgb_image(root / "fabric" / "eval" / "tear" / "002.png", (30, 30, 30))
    _write_mask(root / "fabric" / "masks" / "tear" / "002_gt.png")

    spec = FolderBenchmarkSpec(
        name="custom_fabric",
        train_normal_subdir="normal",
        test_split_dir="eval",
        test_normal_subdir="normal",
        ground_truth_dir="masks",
        mask_suffix="_gt",
    )
    dataset = read_vision_benchmark_dataset(root=root, benchmark=spec, data_mode="paths")

    assert dataset.name() == "CUSTOM_FABRIC-VisionBenchmark"
    assert dataset.train_concepts()[0].data.dtype == object
    assert dataset.test_concepts()[0].data.dtype == object
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([0, 1]))

    samples = index_vision_benchmark(root=root, benchmark=spec)
    assert samples[-1].mask_path == root / "fabric" / "masks" / "tear" / "002_gt.png"


def test_read_vision_benchmark_dataset_supports_custom_csv_spec(tmp_path: Path):
    root = tmp_path / "csv_like"
    _write_rgb_image(root / "part" / "normal" / "train_0.png", (10, 20, 30))
    _write_rgb_image(root / "part" / "normal" / "test_0.png", (20, 30, 40))
    _write_rgb_image(root / "part" / "anomaly" / "test_1.png", (30, 40, 50))
    _write_mask(root / "part" / "masks" / "test_1.png")
    with (root / "splits.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["group", "phase", "target", "img", "seg"])
        writer.writeheader()
        writer.writerow({"group": "part", "phase": "train", "target": "ok", "img": "part/normal/train_0.png", "seg": ""})
        writer.writerow({"group": "part", "phase": "test", "target": "ok", "img": "part/normal/test_0.png", "seg": ""})
        writer.writerow(
            {
                "group": "part",
                "phase": "test",
                "target": "defect",
                "img": "part/anomaly/test_1.png",
                "seg": "part/masks/test_1.png",
            }
        )

    spec = CsvBenchmarkSpec(
        name="custom_csv",
        csv_path="splits.csv",
        category_column="group",
        split_column="phase",
        label_column="target",
        normal_label_value="ok",
        image_column="img",
        mask_column="seg",
    )
    dataset = read_vision_benchmark_dataset(root=root, benchmark=spec, resize_to=(3, 3), color_mode="grayscale")

    assert dataset.train_concepts()[0].data.shape == (1, 3, 3, 1)
    assert dataset.test_concepts()[0].data.shape == (2, 3, 3, 1)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([0, 1]))


def test_read_vision_benchmark_dataset_supports_btech_preset(tmp_path: Path):
    root = tmp_path / "btech_like"
    _write_rgb_image(root / "01" / "train" / "ok" / "000.bmp", (10, 20, 30))
    _write_rgb_image(root / "01" / "test" / "ok" / "001.bmp", (15, 25, 35))
    _write_rgb_image(root / "01" / "test" / "ko" / "002.bmp", (50, 60, 70))
    _write_mask(root / "01" / "ground_truth" / "ko" / "002.png")

    dataset = read_vision_benchmark_dataset(root=root, benchmark="btech", resize_to=(4, 4))

    assert dataset.train_concepts()[0].data.shape == (1, 4, 4, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([1, 0]))

    samples = index_vision_benchmark(root=root, benchmark="btech")
    assert samples[1].mask_path == root / "01" / "ground_truth" / "ko" / "002.png"


def test_read_vision_benchmark_dataset_supports_dagm_preset(tmp_path: Path):
    root = tmp_path / "dagm_like"
    _write_rgb_image(root / "Class1" / "Train" / "000.PNG", (10, 20, 30))
    _write_rgb_image(root / "Class1" / "Train" / "001.PNG", (40, 50, 60))
    _write_mask(root / "Class1" / "Train" / "Label" / "001_label.PNG")
    _write_rgb_image(root / "Class1" / "Test" / "100.PNG", (20, 30, 40))
    _write_rgb_image(root / "Class1" / "Test" / "101.PNG", (50, 60, 70))
    _write_mask(root / "Class1" / "Test" / "Label" / "101_label.PNG")

    dataset = read_vision_benchmark_dataset(root=root, benchmark="dagm", resize_to=(4, 4))

    assert dataset.train_concepts()[0].data.shape == (1, 4, 4, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([0, 1]))

    samples = index_vision_benchmark(root=root, benchmark="dagm")
    assert samples[-1].mask_path is not None
    assert samples[-1].mask_path.name.lower() == "101_label.png"


def test_build_vision_benchmark_reader_returns_dataset_specific_reader(tmp_path: Path):
    expected_types = {
        "btech": BTechBenchmarkReader,
        "dagm": DAGMBenchmarkReader,
        "mpdd": MPDDBenchmarkReader,
        "mvtec": MVTecBenchmarkReader,
        "visa": VisABenchmarkReader,
    }

    for benchmark_name, expected_type in expected_types.items():
        reader = build_vision_benchmark_reader(root=tmp_path, benchmark=benchmark_name)
        assert isinstance(reader, expected_type)
