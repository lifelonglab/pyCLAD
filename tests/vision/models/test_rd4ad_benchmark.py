"""End-to-end accuracy guard for RD4AD on a real dataset (BTech category 01).

This complements the fast unit tests: those prove the wiring and that the loss decreases, but
not that the model actually *detects* anomalies. Here we train on nominal images and assert that
image- and pixel-level ROC-AUC clear a sanity floor well above chance, catching regressions that
keep training/inference running yet quietly destroy detection quality.

The run is slow and data-dependent, so it is opt-in: set ``PYCLAD_RUN_BENCHMARKS=1`` and have the
BTech dataset present under ``examples/resources/vision/BTech_Dataset_transformed``. Otherwise it
skips. A full multi-category sweep lives in ``examples/models/vision/rd4ad_benchmark.py``.
"""

import os
import pathlib

import numpy as np
import pytest

from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.vision.data.readers.vision_reader import read_vision_dataset
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.rd4ad.rd4ad import RD4AD

_BTECH_ROOT = pathlib.Path(__file__).resolve().parents[3] / "examples/resources/vision/BTech_Dataset_transformed"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PYCLAD_RUN_BENCHMARKS") != "1",
        reason="slow end-to-end benchmark; set PYCLAD_RUN_BENCHMARKS=1 to run",
    ),
    pytest.mark.skipif(not _BTECH_ROOT.exists(), reason=f"BTech dataset not found at {_BTECH_ROOT}"),
]


def test_rd4ad_detects_anomalies_on_btech_category():
    dataset = read_vision_dataset(
        root=_BTECH_ROOT,
        benchmark="btech",
        categories=["01"],
        resize_to=(256, 256),
        data_mode="numpy",
        color_mode="rgb",
        max_train_samples_per_category=60,
    )
    train_concept = dataset.train_concepts()[0]
    test_concept = dataset.test_concepts()[0]

    model = RD4AD(
        RD4ADConfig(
            backbone_name="resnet18",
            input_size=(256, 256),
            epochs=20,
            batch_size=16,
            pretrained_encoder=True,
            freeze_encoder=True,
            score_smoothing_sigma=4.0,
            seed=42,
            show_training_progress=False,
        )
    )
    model.fit(train_concept.data)
    result = model.predict(test_concept.data)

    labels = np.asarray(test_concept.labels)
    assert set(np.unique(labels)) == {0, 1}  # both nominal and anomalous present
    # Masks must contain both normal (0) and anomalous (1) pixels, else PixelRocAuc returns nan
    # and the failure would misattribute a degenerate-data problem to a detector regression.
    assert len(np.unique(np.asarray(test_concept.masks))) == 2

    image_roc_auc = RocAuc().compute(result.anomaly_scores, result.y_pred, labels)
    pixel_roc_auc = PixelRocAuc().compute(result.score_maps, result.y_pred, test_concept.masks)

    # Conservative floors: a quick resnet18 run reaches ~0.96 on this category, so 0.75 is a wide
    # margin above chance (0.5) that still catches a genuinely broken detector.
    assert image_roc_auc > 0.75, f"image ROC-AUC too low: {image_roc_auc:.3f}"
    assert pixel_roc_auc > 0.75, f"pixel ROC-AUC too low: {pixel_roc_auc:.3f}"
