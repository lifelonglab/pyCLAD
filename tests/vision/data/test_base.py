from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyclad.vision.data.base import VisionSample, build_concepts_dataset_from_samples
from pyclad.vision.data.vision_concept import VisionConcept


def test_vision_concept_rejects_masks_misaligned_with_data():
    with pytest.raises(ValueError, match="batch dimension"):
        VisionConcept(
            name="widget",
            data=np.zeros((3, 4, 4, 3), dtype=np.float32),
            labels=np.array([0, 1, 0], dtype=np.int64),
            masks=np.zeros((2, 4, 4), dtype=np.uint8),
        )


def test_vision_concept_accepts_aligned_masks():
    concept = VisionConcept(
        name="widget",
        data=np.zeros((2, 4, 4, 3), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        masks=np.zeros((2, 4, 4), dtype=np.uint8),
    )
    assert concept.masks.shape[0] == concept.data.shape[0] == 2


def test_vision_concept_without_masks_skips_alignment_check():
    concept = VisionConcept(
        name="widget",
        data=np.zeros((5, 4, 4, 3), dtype=np.float32),
        labels=np.array([0, 1, 0, 1, 0], dtype=np.int64),
    )
    assert concept.masks is None


def _write_rgb_image(path: Path, rgb: tuple[int, int, int] = (128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), rgb).save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), value).save(path)


def test_build_concepts_dataset_keeps_data_labels_masks_aligned_when_anomaly_mask_missing(tmp_path: Path):
    """Samples skipped from masks must also be dropped from data and labels."""
    _write_rgb_image(tmp_path / "train_normal.png")
    _write_rgb_image(tmp_path / "test_normal.png")
    _write_rgb_image(tmp_path / "test_anomaly_with_mask.png")
    _write_rgb_image(tmp_path / "test_anomaly_no_mask.png")
    _write_mask(tmp_path / "test_anomaly_with_mask_mask.png")

    samples = [
        VisionSample(
            category="widget",
            split="train",
            image_path=tmp_path / "train_normal.png",
            image_label=0,
        ),
        VisionSample(
            category="widget",
            split="test",
            image_path=tmp_path / "test_normal.png",
            image_label=0,
        ),
        VisionSample(
            category="widget",
            split="test",
            image_path=tmp_path / "test_anomaly_with_mask.png",
            image_label=1,
            mask_path=tmp_path / "test_anomaly_with_mask_mask.png",
        ),
        VisionSample(
            category="widget",
            split="test",
            image_path=tmp_path / "test_anomaly_no_mask.png",
            image_label=1,
            mask_path=None,
        ),
    ]

    dataset = build_concepts_dataset_from_samples(samples=samples, dataset_name="missing-mask")

    test_concept = dataset.test_concepts()[0]
    assert isinstance(test_concept, VisionConcept)
    assert test_concept.data.shape[0] == test_concept.labels.shape[0] == test_concept.masks.shape[0]
    assert test_concept.data.shape[0] == 2


def test_build_concepts_dataset_returns_plain_concept_when_no_masks(tmp_path: Path):
    _write_rgb_image(tmp_path / "train.png")
    _write_rgb_image(tmp_path / "test.png")

    samples = [
        VisionSample(category="widget", split="train", image_path=tmp_path / "train.png", image_label=0),
        VisionSample(category="widget", split="test", image_path=tmp_path / "test.png", image_label=0),
    ]

    dataset = build_concepts_dataset_from_samples(samples=samples, dataset_name="no-mask")
    test_concept = dataset.test_concepts()[0]

    assert not isinstance(test_concept, VisionConcept)
    assert test_concept.data.shape[0] == test_concept.labels.shape[0] == 1
