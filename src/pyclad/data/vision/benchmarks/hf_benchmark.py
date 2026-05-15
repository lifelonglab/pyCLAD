from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from datasets import load_dataset

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.vision.benchmarks.manifest import (
    VISION_BENCHMARK_MANIFEST_FIELDNAMES,
    build_vision_benchmark_manifest_spec,
)
from pyclad.data.vision.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.data.vision.benchmarks.registry import resolve_vision_benchmark_root

INCLAD_BENCH_HF_REPO = "anonmllab/inclad-bench"

INCLAD_BENCH_CONFIGS = ("btech", "dagm", "mpdd", "mvtec", "visa")

INCLAD_BENCH_ORDERINGS = ("easy_to_hard", "hard_to_easy", "random")

InCLADBenchConfig = Literal["btech", "dagm", "mpdd", "mvtec", "visa"]
InCLADBenchOrdering = Literal["easy_to_hard", "hard_to_easy", "random"]


def _hf_manifest_to_csv(hf_dataset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = hf_dataset.to_pandas()

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VISION_BENCHMARK_MANIFEST_FIELDNAMES)
        writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow({field: str(row.get(field, "")) for field in VISION_BENCHMARK_MANIFEST_FIELDNAMES})
    return output_path


class InCLADBenchDataset(ConceptsDataset):
    """
    Loader for the InCLAD-Bench benchmark hosted on HuggingFace
    (https://huggingface.co/datasets/anonmllab/inclad-bench).

    The HuggingFace dataset contains sample manifests (metadata with image
    relative paths, labels, and category orderings) for five vision anomaly
    detection benchmarks: BTech, DAGM, MPDD, MVTec AD, and VisA.

    Actual images are *not* included in the HuggingFace repository — they
    must be available locally. The ``root`` parameter (or the pyCLAD vision
    dataset registry / environment variables) is used to locate the images
    on disk.
    """

    def __init__(
        self,
        benchmark: InCLADBenchConfig,
        ordering: InCLADBenchOrdering = "easy_to_hard",
        root: Optional[Union[str, Path]] = None,
        registry_path: Optional[Union[str, Path]] = None,
        categories: Optional[Sequence[str]] = None,
        data_mode: str = "numpy",
        resize_to: Optional[tuple[int, int]] = None,
        color_mode: str = "rgb",
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        :param benchmark: One of ``"btech"``, ``"dagm"``, ``"mpdd"``, ``"mvtec"``, ``"visa"``.
        :param ordering: Category ordering strategy — ``"easy_to_hard"``,
            ``"hard_to_easy"``, or ``"random"``.
        :param root: Local path to the benchmark image directory. When ``None``,
            resolved via ``PYCLAD_VISION_DATASETS_ROOT``, per-benchmark env vars,
            or the pyCLAD vision dataset registry.
        :param registry_path: Optional path to the vision dataset registry JSON.
        :param categories: Subset of categories to include. ``None`` = all.
        :param data_mode: ``"numpy"`` to load images as arrays, ``"paths"`` for
            file paths only.
        :param resize_to: ``(height, width)`` to resize images. ``None`` keeps
            original size.
        :param color_mode: ``"rgb"`` or ``"grayscale"``.
        :param max_train_samples_per_category: Cap on training samples per category.
        :param max_test_samples_per_category: Cap on test samples per category.
        :param cache_dir: HuggingFace cache directory. ``None`` uses the default.
        """
        benchmark_key = benchmark.lower()
        if benchmark_key not in INCLAD_BENCH_CONFIGS:
            raise ValueError(f"Unknown InCLAD-Bench benchmark '{benchmark}'. " f"Available: {INCLAD_BENCH_CONFIGS}")
        if ordering not in INCLAD_BENCH_ORDERINGS:
            raise ValueError(f"Unknown ordering '{ordering}'. Available: {INCLAD_BENCH_ORDERINGS}")

        resolved_root = resolve_vision_benchmark_root(benchmark=benchmark_key, root=root, registry_path=registry_path)

        hf_dataset = load_dataset(INCLAD_BENCH_HF_REPO, benchmark_key, split=ordering, cache_dir=cache_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / f"{benchmark_key}_{ordering}_manifest.csv"
            _hf_manifest_to_csv(hf_dataset, csv_path)

            spec = build_vision_benchmark_manifest_spec(benchmark=benchmark_key, csv_path=csv_path)
            dataset = read_vision_benchmark_dataset(
                root=resolved_root,
                benchmark=spec,
                dataset_name=f"InCLAD-{benchmark_key}-{ordering}",
                categories=categories,
                data_mode=data_mode,
                resize_to=resize_to,
                color_mode=color_mode,
                max_train_samples_per_category=max_train_samples_per_category,
                max_test_samples_per_category=max_test_samples_per_category,
            )

        super().__init__(
            name=dataset.name(),
            train_concepts=dataset.train_concepts(),
            test_concepts=dataset.test_concepts(),
        )


def load_inclad_bench(
    benchmark: InCLADBenchConfig,
    ordering: InCLADBenchOrdering = "easy_to_hard",
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> ConceptsDataset:
    """Convenience function to load an InCLAD-Bench dataset.

    See :class:`InCLADBenchDataset` for parameter documentation.
    """
    return InCLADBenchDataset(
        benchmark=benchmark,
        ordering=ordering,
        root=root,
        registry_path=registry_path,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        cache_dir=cache_dir,
    )
