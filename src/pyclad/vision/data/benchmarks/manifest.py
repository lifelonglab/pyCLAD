from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data._utils import find_duplicates, infer_category_order_from_samples
from pyclad.vision.data.benchmarks.ordering import (
    load_concept_order_from_file,
    resolve_benchmark_ordering_path,
)
from pyclad.vision.data.benchmarks.readers import (
    PREDEFINED_BENCHMARK_ALIASES,
    CsvBenchmarkSpec,
    VisionSample,
    index_vision_benchmark,
    read_vision_benchmark_dataset,
)

VISION_BENCHMARK_MANIFEST_FIELDNAMES = (
    "sample_id",
    "source_dataset",
    "category",
    "category_order",
    "split",
    "image_relpath",
    "mask_relpath",
    "image_label",
    "defect_type",
    "ordering_name",
    "ordering_master_seed",
    "ordering_seed",
    "source_homepage",
    "source_license",
)

VISION_BENCHMARK_SOURCES = {
    "btech": {
        "homepage": "https://github.com/pankajmishra000/VT-ADL",
        "license": "CC BY-SA",
    },
    "dagm": {
        "homepage": "https://zenodo.org/records/12750201",
        "license": "CC BY 4.0",
    },
    "mpdd": {
        "homepage": "https://github.com/stepanje/MPDD",
        "license": "CC BY-NC-SA 4.0",
    },
    "mvtec": {
        "homepage": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "license": "CC BY-NC-SA 4.0",
    },
    "visa": {
        "homepage": "https://github.com/amazon-science/spot-diff",
        "license": "CC BY 4.0",
    },
}


@dataclass(frozen=True)
class VisionBenchmarkManifestOrdering:
    name: str
    category_order: list[str]
    master_seed: Optional[int] = None
    seed: Optional[int] = None


def build_vision_benchmark_manifest_spec(
    benchmark: str,
    csv_path: Union[str, Path],
) -> CsvBenchmarkSpec:
    benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
    resolved_csv_path = Path(csv_path).expanduser().resolve()
    return CsvBenchmarkSpec(
        name=benchmark_name,
        csv_path=str(resolved_csv_path),
        category_column="category",
        category_order_column="category_order",
        split_column="split",
        train_split_value="train",
        test_split_value="test",
        image_column="image_relpath",
        label_column="image_label",
        normal_label_value="0",
        mask_column="mask_relpath",
        defect_type_column="defect_type",
    )


def manifest_output_filename(benchmark: str, ordering_name: str = "dataset") -> str:
    benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
    if ordering_name == "dataset":
        return f"{benchmark_name}_samples.csv"
    return f"{benchmark_name}_{ordering_name}_samples.csv"


def resolve_vision_benchmark_manifest_path(
    benchmark: str,
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    ordering_name: str = "dataset",
    ordering_master_seed: Optional[int] = None,
    ordering_seed: Optional[int] = None,
) -> Path:
    if output_path is not None:
        return Path(output_path).expanduser().resolve()

    if output_dir is None:
        raise ValueError("Provide output_path= (full file path) or output_dir= (directory to write the manifest into).")

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    return resolved_output_dir / manifest_output_filename(benchmark=benchmark, ordering_name=ordering_name)


def write_vision_benchmark_manifest(
    benchmark: str,
    root: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    category_order: Optional[Sequence[str]] = None,
    ordering_name: str = "dataset",
    ordering_master_seed: Optional[int] = None,
    ordering_seed: Optional[int] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> Path:
    benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {resolved_root}")
    samples = index_vision_benchmark(
        root=resolved_root,
        benchmark=benchmark_name,
        categories=categories,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )
    resolved_category_order = _resolve_category_order_for_manifest(samples=samples, category_order=category_order)
    ordered_samples = reorder_samples_by_category_order(samples=samples, category_order=resolved_category_order)
    manifest_path = resolve_vision_benchmark_manifest_path(
        benchmark=benchmark_name,
        output_path=output_path,
        output_dir=output_dir,
        ordering_name=ordering_name,
        ordering_master_seed=ordering_master_seed,
        ordering_seed=ordering_seed,
    )
    write_vision_benchmark_manifest_rows(
        benchmark=benchmark_name,
        root=resolved_root,
        samples=ordered_samples,
        output_path=manifest_path,
        category_order=resolved_category_order,
        ordering_name=ordering_name,
        ordering_master_seed=ordering_master_seed,
        ordering_seed=ordering_seed,
    )
    return manifest_path


def write_vision_benchmark_manifests(
    benchmarks: Sequence[str],
    roots: Mapping[str, Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    categories_by_benchmark: Optional[Mapping[str, Sequence[str]]] = None,
    category_order_by_benchmark: Optional[Mapping[str, Sequence[str]]] = None,
    ordering_name: str = "dataset",
    ordering_master_seed: Optional[int] = None,
    ordering_seed: Optional[int] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> dict[str, Path]:
    manifest_paths: dict[str, Path] = {}
    for benchmark in benchmarks:
        benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
        if benchmark_name not in roots:
            raise KeyError(f"roots= must contain an entry for benchmark '{benchmark_name}'")
        benchmark_categories = None if categories_by_benchmark is None else categories_by_benchmark.get(benchmark_name)
        benchmark_category_order = (
            None if category_order_by_benchmark is None else category_order_by_benchmark.get(benchmark_name)
        )
        manifest_paths[benchmark_name] = write_vision_benchmark_manifest(
            benchmark=benchmark_name,
            root=roots[benchmark_name],
            output_dir=output_dir,
            categories=benchmark_categories,
            category_order=benchmark_category_order,
            ordering_name=ordering_name,
            ordering_master_seed=ordering_master_seed,
            ordering_seed=ordering_seed,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
        )
    return manifest_paths


def write_vision_benchmark_manifest_rows(
    benchmark: str,
    root: Union[str, Path],
    samples: Sequence[VisionSample],
    output_path: Union[str, Path],
    category_order: Sequence[str],
    ordering_name: str = "dataset",
    ordering_master_seed: Optional[int] = None,
    ordering_seed: Optional[int] = None,
) -> Path:
    benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
    resolved_root = Path(root).expanduser().resolve()
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    category_positions = {category: index + 1 for index, category in enumerate(category_order)}

    with resolved_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=VISION_BENCHMARK_MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row_index, sample in enumerate(samples):
            writer.writerow(
                build_vision_benchmark_manifest_row(
                    benchmark=benchmark_name,
                    root=resolved_root,
                    sample=sample,
                    row_index=row_index,
                    category_positions=category_positions,
                    ordering_name=ordering_name,
                    ordering_master_seed=ordering_master_seed,
                    ordering_seed=ordering_seed,
                )
            )

    return resolved_output_path


def build_vision_benchmark_manifest_row(
    benchmark: str,
    root: Union[str, Path],
    sample: VisionSample,
    row_index: int,
    category_positions: Mapping[str, int],
    ordering_name: str = "dataset",
    ordering_master_seed: Optional[int] = None,
    ordering_seed: Optional[int] = None,
) -> dict[str, str]:
    benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
    resolved_root = Path(root).expanduser().resolve()
    source_info = VISION_BENCHMARK_SOURCES.get(benchmark_name, {})

    return {
        "sample_id": f"{benchmark_name}:{row_index:06d}",
        "source_dataset": benchmark_name,
        "category": sample.category,
        "category_order": str(category_positions[sample.category]),
        "split": sample.split,
        "image_relpath": _path_relative_to_root(sample.image_path, resolved_root),
        "mask_relpath": _path_relative_to_root(sample.mask_path, resolved_root),
        "image_label": str(int(sample.image_label)),
        "defect_type": sample.defect_type or "",
        "ordering_name": ordering_name,
        "ordering_master_seed": "" if ordering_master_seed is None else str(int(ordering_master_seed)),
        "ordering_seed": "" if ordering_seed is None else str(int(ordering_seed)),
        "source_homepage": str(source_info.get("homepage", "")),
        "source_license": str(source_info.get("license", "")),
    }


def read_vision_benchmark_manifest_dataset(
    root: Union[str, Path],
    manifest_path: Union[str, Path],
    benchmark: str,
    dataset_name: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    spec = build_vision_benchmark_manifest_spec(benchmark=benchmark, csv_path=manifest_path)
    return read_vision_benchmark_dataset(
        root=root,
        benchmark=spec,
        dataset_name=dataset_name,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def index_vision_benchmark_manifest(
    root: Union[str, Path],
    manifest_path: Union[str, Path],
    benchmark: str,
    categories: Optional[Sequence[str]] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> list[VisionSample]:
    spec = build_vision_benchmark_manifest_spec(benchmark=benchmark, csv_path=manifest_path)
    return index_vision_benchmark(
        root=root,
        benchmark=spec,
        categories=categories,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def load_vision_benchmark_manifest_ordering(
    benchmark: str,
    root: Union[str, Path],
    ordering_name: str = "dataset",
    ordering_dir: Optional[Union[str, Path]] = None,
    ordering_metric: str = "roc-auc",
    master_seed: Optional[int] = None,
    run_index: int = 0,
) -> VisionBenchmarkManifestOrdering:
    benchmark_name = normalize_vision_benchmark_manifest_name(benchmark)
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {resolved_root}")
    samples = index_vision_benchmark(root=resolved_root, benchmark=benchmark_name)
    base_category_order = infer_category_order_from_samples(samples)

    if ordering_name == "dataset":
        return VisionBenchmarkManifestOrdering(name="dataset", category_order=base_category_order)

    if ordering_name in {"easy_to_hard", "hard_to_easy"}:
        if ordering_dir is None:
            raise ValueError(
                f"ordering_name={ordering_name!r} requires ordering_dir= pointing to the directory with "
                f"ordering CSV files (e.g. <benchmark>_{ordering_name}_<metric>.csv)."
            )
        resolved_ordering_dir = Path(ordering_dir).expanduser().resolve()
        ordering_path = resolve_benchmark_ordering_path(
            ordering_dir=resolved_ordering_dir,
            benchmark=benchmark_name,
            ordering_mode=ordering_name,
            ordering_metric=ordering_metric,
        )
        category_order = load_concept_order_from_file(
            ordering_path=ordering_path,
            benchmark=benchmark_name,
            ordering_column="concept",
            ordering_mode=ordering_name,
        )
        _validate_category_order(category_order, base_category_order)
        seed = None if master_seed is None else derive_per_run_seed(master_seed=master_seed, run_index=run_index)
        return VisionBenchmarkManifestOrdering(
            name=ordering_name,
            category_order=list(category_order),
            master_seed=master_seed,
            seed=seed,
        )

    if ordering_name == "random":
        if master_seed is None:
            raise ValueError("random ordering requires master_seed")
        seed = derive_per_run_seed(master_seed=master_seed, run_index=run_index)
        return VisionBenchmarkManifestOrdering(
            name="random",
            category_order=shuffle_category_order(base_category_order, seed=seed),
            master_seed=master_seed,
            seed=seed,
        )

    raise ValueError(f"Unsupported ordering_name: {ordering_name!r}")


def derive_per_run_seed(master_seed: int, run_index: int = 0) -> int:
    if run_index < 0:
        raise ValueError(f"run_index must be >= 0, got {run_index}")
    rng = np.random.default_rng(int(master_seed))
    seeds = rng.integers(0, 2**31 - 1, size=run_index + 1)
    return int(seeds[run_index])


def reorder_samples_by_category_order(
    samples: Sequence[VisionSample],
    category_order: Sequence[str],
) -> list[VisionSample]:
    resolved_order = _resolve_category_order_for_manifest(samples=samples, category_order=category_order)
    order_positions = {category: index for index, category in enumerate(resolved_order)}
    return sorted(
        samples,
        key=lambda sample: (
            order_positions[sample.category],
            0 if sample.split == "train" else 1,
            str(sample.image_path),
        ),
    )


def shuffle_category_order(category_order: Sequence[str], seed: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    category_list = list(category_order)
    indices = rng.permutation(len(category_list))
    return [category_list[index] for index in indices]


def normalize_vision_benchmark_manifest_name(benchmark: str) -> str:
    key = str(benchmark).lower()
    return PREDEFINED_BENCHMARK_ALIASES.get(key, key)


def _resolve_category_order_for_manifest(
    samples: Sequence[VisionSample],
    category_order: Optional[Sequence[str]],
) -> list[str]:
    available_categories = infer_category_order_from_samples(samples)
    if category_order is None:
        return available_categories
    resolved_category_order = [str(category) for category in category_order]
    _validate_category_order(resolved_category_order, available_categories)
    return resolved_category_order


def _validate_category_order(category_order: Sequence[str], available_categories: Sequence[str]) -> None:
    normalized_available = list(available_categories)
    duplicates = find_duplicates(category_order)
    if duplicates:
        raise ValueError(f"Category order contains duplicates: {duplicates}")

    missing = [category for category in normalized_available if category not in category_order]
    if missing:
        raise ValueError(f"Category order does not fully cover the benchmark categories. Missing categories: {missing}")

    extra = [category for category in category_order if category not in normalized_available]
    if extra:
        raise ValueError(f"Category order contains unknown categories: {extra}")


def _path_relative_to_root(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""

    resolved_path = Path(path).expanduser().resolve()
    try:
        return str(resolved_path.relative_to(root))
    except ValueError:
        return str(resolved_path)
