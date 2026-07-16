"""End-to-end accuracy benchmark for the RD4AD vision model.

Unlike the unit tests (which only prove the wiring and that the loss decreases), this script
actually trains RD4AD on nominal images and reports image- and pixel-level detection quality on
the held-out test split, so you get a real reproducibility datapoint to compare against the paper.

It is a plain script, not a pytest test, because a full run is slow and data-dependent. A guarded,
fast version lives in ``tests/vision/models/test_rd4ad_benchmark.py``.

Example:
    .venv/bin/python examples/models/vision/rd4ad_benchmark.py --categories 01 --epochs 50
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Optional

import numpy as np

from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.vision.data.readers.vision_reader import read_vision_dataset
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.rd4ad.rd4ad import RD4AD

_DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources/vision/BTech_Dataset_transformed"


def evaluate_category(model: RD4AD, test_concept) -> dict:
    """Train-free evaluation of an already-fitted model on one test concept."""
    result = model.predict(test_concept.data)
    labels = np.asarray(test_concept.labels)

    metrics = {"image_roc_auc": float(RocAuc().compute(result.anomaly_scores, result.y_pred, labels))}

    masks = getattr(test_concept, "masks", None)
    if masks is not None:
        metrics["pixel_roc_auc"] = float(PixelRocAuc().compute(result.score_maps, result.y_pred, masks))
        metrics["pixel_aupro"] = float(PixelAUPRO().compute(result.score_maps, result.y_pred, masks))

    return metrics


def run_benchmark(
    root: pathlib.Path,
    benchmark: Optional[str],
    categories: Optional[list[str]],
    backbone: str,
    input_size: tuple[int, int],
    epochs: int,
    batch_size: int,
    smoothing_sigma: float,
    max_train: Optional[int],
    max_test: Optional[int],
    device: Optional[str],
    seed: int,
) -> dict[str, dict]:
    dataset = read_vision_dataset(
        root=root,
        benchmark=benchmark,
        categories=categories,
        resize_to=input_size,
        data_mode="numpy",
        color_mode="rgb",
        max_train_samples_per_category=max_train,
        max_test_samples_per_category=max_test,
    )

    train_by_name = {concept.name: concept for concept in dataset.train_concepts()}
    results: dict[str, dict] = {}

    for test_concept in dataset.test_concepts():
        name = test_concept.name
        train_concept = train_by_name[name]

        model = RD4AD(
            RD4ADConfig(
                backbone_name=backbone,
                input_size=input_size,
                epochs=epochs,
                batch_size=batch_size,
                pretrained_encoder=True,
                freeze_encoder=True,
                score_smoothing_sigma=smoothing_sigma,
                device=device,
                seed=seed,
                show_training_progress=False,
            )
        )

        start = time.perf_counter()
        model.fit(train_concept.data)
        metrics = evaluate_category(model, test_concept)
        metrics["fit_predict_seconds"] = round(time.perf_counter() - start, 1)
        metrics["n_train"] = int(len(train_concept.data))
        metrics["n_test"] = int(len(test_concept.data))
        results[name] = metrics

        print(
            f"[{name}] image-AUROC={metrics['image_roc_auc']:.3f} "
            f"pixel-AUROC={metrics.get('pixel_roc_auc', float('nan')):.3f} "
            f"pixel-AUPRO={metrics.get('pixel_aupro', float('nan')):.3f} "
            f"(train={metrics['n_train']} test={metrics['n_test']} {metrics['fit_predict_seconds']}s)"
        )

    if results:
        for key in ("image_roc_auc", "pixel_roc_auc", "pixel_aupro"):
            values = [m[key] for m in results.values() if key in m and not np.isnan(m[key])]
            if values:
                print(f"mean {key}: {np.mean(values):.3f}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RD4AD end-to-end accuracy benchmark")
    parser.add_argument("--root", type=pathlib.Path, default=_DEFAULT_ROOT)
    parser.add_argument("--benchmark", type=str, default="btech")
    parser.add_argument("--categories", nargs="*", default=None, help="subset of categories (default: all)")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--input-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoothing-sigma", type=float, default=4.0, help="paper uses 4.0; 0 disables")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        root=args.root,
        benchmark=args.benchmark,
        categories=args.categories,
        backbone=args.backbone,
        input_size=tuple(args.input_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoothing_sigma=args.smoothing_sigma,
        max_train=args.max_train,
        max_test=args.max_test,
        device=args.device,
        seed=args.seed,
    )
