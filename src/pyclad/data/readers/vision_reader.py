"""Public entrypoints for vision dataset readers.

This module re-exports the shared reader primitives and provides a single
dispatch layer that builds either a generic folder reader or a benchmark
reader depending on the given arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Optional, Sequence, Tuple, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.vision.base import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VisionBenchmarkReader,
    VisionSample,
    build_concepts_dataset_from_samples,
    list_image_files,
    materialize_samples,
    select_categories,
    validate_read_options,
)
from pyclad.data.vision.benchmarks.readers import VisionBenchmarkSpec
from pyclad.data.vision.benchmarks.registry import (
    build_registered_vision_benchmark_reader,
)
from pyclad.data.vision.generic_reader import (
    DEFAULT_NORMAL_LABELS,
    FolderLayout,
    GenericFolderReader,
)


def build_vision_reader(
    root: Optional[Union[str, Path]] = None,
    benchmark: Optional[Union[str, VisionBenchmarkSpec]] = None,
    name: str = "custom",
    layout: Optional[Union[str, FolderLayout]] = None,
    train_split_dir: str = "train",
    test_split_dir: str = "test",
    normal_labels: FrozenSet[str] = DEFAULT_NORMAL_LABELS,
    ground_truth_dir: Optional[str] = "ground_truth",
    mask_suffix: str = "_mask",
    image_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    single_category_name: str = "default",
    registry_path: Optional[Union[str, Path]] = None,
) -> VisionBenchmarkReader:
    if benchmark is not None:
        return build_registered_vision_benchmark_reader(
            benchmark=benchmark,
            root=root,
            registry_path=registry_path,
        )

    if root is None:
        raise ValueError("root is required when benchmark is not provided")

    resolved_layout = layout if layout is not None else GenericFolderReader.detect_layout(root)
    return GenericFolderReader(
        root=root,
        name=name,
        layout=resolved_layout,
        train_split_dir=train_split_dir,
        test_split_dir=test_split_dir,
        normal_labels=normal_labels,
        ground_truth_dir=ground_truth_dir,
        mask_suffix=mask_suffix,
        image_extensions=image_extensions,
        single_category_name=single_category_name,
    )


def index_vision_dataset(
    root: Optional[Union[str, Path]] = None,
    benchmark: Optional[Union[str, VisionBenchmarkSpec]] = None,
    name: str = "custom",
    layout: Optional[Union[str, FolderLayout]] = None,
    train_split_dir: str = "train",
    test_split_dir: str = "test",
    normal_labels: FrozenSet[str] = DEFAULT_NORMAL_LABELS,
    ground_truth_dir: Optional[str] = "ground_truth",
    mask_suffix: str = "_mask",
    image_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    single_category_name: str = "default",
    registry_path: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> list[VisionSample]:
    reader = build_vision_reader(
        root=root,
        benchmark=benchmark,
        name=name,
        layout=layout,
        train_split_dir=train_split_dir,
        test_split_dir=test_split_dir,
        normal_labels=normal_labels,
        ground_truth_dir=ground_truth_dir,
        mask_suffix=mask_suffix,
        image_extensions=image_extensions,
        single_category_name=single_category_name,
        registry_path=registry_path,
    )
    return reader.index_samples(
        categories=categories,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def read_vision_dataset(
    root: Optional[Union[str, Path]] = None,
    benchmark: Optional[Union[str, VisionBenchmarkSpec]] = None,
    name: str = "custom",
    layout: Optional[Union[str, FolderLayout]] = None,
    train_split_dir: str = "train",
    test_split_dir: str = "test",
    normal_labels: FrozenSet[str] = DEFAULT_NORMAL_LABELS,
    ground_truth_dir: Optional[str] = "ground_truth",
    mask_suffix: str = "_mask",
    image_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    single_category_name: str = "default",
    registry_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    reader = build_vision_reader(
        root=root,
        benchmark=benchmark,
        name=name,
        layout=layout,
        train_split_dir=train_split_dir,
        test_split_dir=test_split_dir,
        normal_labels=normal_labels,
        ground_truth_dir=ground_truth_dir,
        mask_suffix=mask_suffix,
        image_extensions=image_extensions,
        single_category_name=single_category_name,
        registry_path=registry_path,
    )
    return reader.read_dataset(
        dataset_name=dataset_name,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


__all__ = [
    "DEFAULT_NORMAL_LABELS",
    "FolderLayout",
    "GenericFolderReader",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "VisionBenchmarkReader",
    "VisionSample",
    "build_concepts_dataset_from_samples",
    "build_vision_reader",
    "index_vision_dataset",
    "list_image_files",
    "materialize_samples",
    "read_vision_dataset",
    "select_categories",
    "validate_read_options",
]
