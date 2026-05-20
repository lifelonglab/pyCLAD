import csv
from pathlib import Path

import numpy as np
from PIL import Image

from pyclad.vision.data.benchmarks.manifest import (
    VISION_BENCHMARK_MANIFEST_FIELDNAMES,
    build_vision_benchmark_manifest_spec,
    derive_per_run_seed,
    index_vision_benchmark_manifest,
    load_vision_benchmark_manifest_ordering,
    manifest_output_filename,
    read_vision_benchmark_manifest_dataset,
    write_registered_vision_benchmark_manifest,
)


def _write_rgb_image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((6, 5, 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, value: int = 255):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((6, 5), value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_write_registered_vision_benchmark_manifest_exports_standard_csv(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png")

    manifest_path = write_registered_vision_benchmark_manifest(
        benchmark="mvtec",
        root=root,
        output_dir=tmp_path / "manifests",
    )

    assert manifest_path.name == manifest_output_filename("mvtec")

    with manifest_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert tuple(rows[0].keys()) == VISION_BENCHMARK_MANIFEST_FIELDNAMES
    assert rows[0]["sample_id"] == "mvtec:000000"
    assert rows[0]["source_dataset"] == "mvtec"
    assert rows[0]["category"] == "widget"
    assert rows[0]["category_order"] == "1"
    assert rows[0]["split"] == "train"
    assert rows[0]["image_relpath"] == "widget/train/good/000.png"
    assert rows[0]["mask_relpath"] == ""
    assert rows[0]["image_label"] == "0"
    assert rows[0]["defect_type"] == ""
    assert rows[0]["ordering_name"] == "dataset"
    assert rows[0]["ordering_master_seed"] == ""
    assert rows[0]["ordering_seed"] == ""
    assert rows[0]["source_homepage"]
    assert rows[0]["source_license"]

    anomaly_row = next(row for row in rows if row["image_label"] == "1")
    assert anomaly_row["split"] == "test"
    assert anomaly_row["defect_type"] == "crack"
    assert anomaly_row["mask_relpath"] == "widget/ground_truth/crack/101_mask.png"


def test_read_vision_benchmark_manifest_dataset_round_trips_back_into_dataset(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png")

    manifest_path = write_registered_vision_benchmark_manifest(
        benchmark="mvtec",
        root=root,
        output_path=tmp_path / "mvtec_samples.csv",
    )

    dataset = read_vision_benchmark_manifest_dataset(
        root=root,
        manifest_path=manifest_path,
        benchmark="mvtec",
        data_mode="paths",
    )
    samples = index_vision_benchmark_manifest(root=root, manifest_path=manifest_path, benchmark="mvtec")

    assert dataset.train_concepts()[0].data.dtype == object
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([1, 0]))
    anomaly_sample = next(sample for sample in samples if sample.image_label == 1)
    assert anomaly_sample.defect_type == "crack"
    assert anomaly_sample.mask_path == root / "widget" / "ground_truth" / "crack" / "101_mask.png"


def test_build_vision_benchmark_manifest_spec_uses_standard_columns():
    spec = build_vision_benchmark_manifest_spec(benchmark="mvtec_ad", csv_path="/tmp/mvtec_samples.csv")

    assert spec.name == "mvtec"
    assert spec.category_column == "category"
    assert spec.category_order_column == "category_order"
    assert spec.split_column == "split"
    assert spec.image_column == "image_relpath"
    assert spec.label_column == "image_label"
    assert spec.normal_label_value == "0"
    assert spec.mask_column == "mask_relpath"
    assert spec.defect_type_column == "defect_type"


def test_manifest_output_filename_keeps_seeds_in_columns_not_in_filename():
    assert manifest_output_filename("mvtec", ordering_name="dataset") == "mvtec_samples.csv"
    assert (
        manifest_output_filename(
            "mvtec",
            ordering_name="easy_to_hard",
            ordering_master_seed=42,
            ordering_seed=191664963,
        )
        == "mvtec_easy_to_hard_samples.csv"
    )
    assert (
        manifest_output_filename(
            "mvtec",
            ordering_name="random",
            ordering_master_seed=123,
            ordering_seed=derive_per_run_seed(123),
        )
        == "mvtec_random_samples.csv"
    )


def test_read_vision_benchmark_manifest_dataset_preserves_manifest_category_order(tmp_path: Path):
    root = tmp_path / "btech_like"
    for category, base in (("01", 10), ("02", 40), ("03", 70)):
        _write_rgb_image(root / category / "train" / "ok" / "000.bmp", (base, base + 1, base + 2))
        _write_rgb_image(root / category / "test" / "ok" / "001.bmp", (base + 3, base + 4, base + 5))
        _write_rgb_image(root / category / "test" / "ko" / "002.bmp", (base + 6, base + 7, base + 8))
        _write_mask(root / category / "ground_truth" / "ko" / "002.png")

    manifest_path = write_registered_vision_benchmark_manifest(
        benchmark="btech",
        root=root,
        output_path=tmp_path / "btech_easy_to_hard.csv",
        category_order=["03", "01", "02"],
        ordering_name="easy_to_hard",
        ordering_master_seed=42,
        ordering_seed=191664963,
    )

    dataset = read_vision_benchmark_manifest_dataset(
        root=root,
        manifest_path=manifest_path,
        benchmark="btech",
        data_mode="paths",
    )

    assert [concept.name for concept in dataset.train_concepts()] == ["03", "01", "02"]

    with manifest_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    first_category_rows = [row for row in rows if row["category"] == "03"]
    assert first_category_rows
    assert all(row["category_order"] == "1" for row in first_category_rows)
    assert all(row["ordering_master_seed"] == "42" for row in rows)
    assert all(row["ordering_seed"] == "191664963" for row in rows)


def test_load_vision_benchmark_manifest_ordering_supports_random_seed_derivation(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    for category, base in (("bottle", 10), ("capsule", 40), ("hazelnut", 70)):
        _write_rgb_image(root / category / "train" / "good" / "000.png", (base, base + 1, base + 2))
        _write_rgb_image(root / category / "test" / "good" / "001.png", (base + 3, base + 4, base + 5))

    ordering = load_vision_benchmark_manifest_ordering(
        benchmark="mvtec",
        root=root,
        ordering_name="random",
        master_seed=123,
    )

    assert ordering.master_seed == 123
    assert ordering.seed == derive_per_run_seed(123)
    assert sorted(ordering.category_order) == ["bottle", "capsule", "hazelnut"]
