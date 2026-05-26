from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from datasets import load_dataset

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data.benchmarks.manifest import (
    VISION_BENCHMARK_MANIFEST_FIELDNAMES,
    build_vision_benchmark_manifest_spec,
)
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset

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
    """Loader for the InCLAD-Bench benchmark hosted on HuggingFace
    (https://huggingface.co/datasets/anonmllab/inclad-bench).

    The HuggingFace dataset contains only sample manifests (metadata with image
    relative paths, labels, and category orderings) for five vision anomaly
    detection benchmarks: BTech, DAGM, MPDD, MVTec AD, and VisA. The actual
    images are *not* in the HuggingFace repo — pass ``root=`` pointing at the
    local benchmark directory containing them.
    """

    def __init__(
        self,
        benchmark: InCLADBenchConfig,
        root: Union[str, Path],
        ordering: InCLADBenchOrdering = "easy_to_hard",
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
        :param root: Local path to the benchmark image directory.
        :param ordering: Category ordering strategy — ``"easy_to_hard"``,
            ``"hard_to_easy"``, or ``"random"``.
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
            raise ValueError(f"Unknown InCLAD-Bench benchmark '{benchmark}'. Available: {INCLAD_BENCH_CONFIGS}")
        if ordering not in INCLAD_BENCH_ORDERINGS:
            raise ValueError(f"Unknown ordering '{ordering}'. Available: {INCLAD_BENCH_ORDERINGS}")

        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(f"InCLAD-Bench root does not exist: {resolved_root}")

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
    root: Union[str, Path],
    ordering: InCLADBenchOrdering = "easy_to_hard",
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> ConceptsDataset:
    """Convenience wrapper for :class:`InCLADBenchDataset`."""
    return InCLADBenchDataset(
        benchmark=benchmark,
        root=root,
        ordering=ordering,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        cache_dir=cache_dir,
    )
