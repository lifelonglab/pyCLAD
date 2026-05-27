from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data.base import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VisionBenchmarkReader,
    VisionSample,
    list_image_files,
    select_categories,
)


@dataclass(frozen=True)
class FolderBenchmarkSpec:
    name: str
    train_split_dir: str = "train"
    test_split_dir: str = "test"
    train_normal_subdir: str = "good"
    test_normal_subdir: str = "good"
    ground_truth_dir: str = "ground_truth"
    mask_suffix: str = "_mask"
    image_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class CsvBenchmarkSpec:
    name: str
    csv_path: str
    category_column: str = "object"
    category_order_column: Optional[str] = None
    split_column: str = "split"
    train_split_value: str = "train"
    test_split_value: str = "test"
    image_column: str = "image"
    label_column: str = "label"
    normal_label_value: str = "normal"
    mask_column: Optional[str] = "mask"
    defect_type_column: Optional[str] = None


VisionBenchmarkSpec = Union[FolderBenchmarkSpec, CsvBenchmarkSpec]


class FolderBenchmarkReader(VisionBenchmarkReader):
    def __init__(
        self,
        root: Union[str, Path],
        name: str,
        train_split_dir: str = "train",
        test_split_dir: str = "test",
        train_normal_subdir: str = "good",
        test_normal_subdir: str = "good",
        ground_truth_dir: str = "ground_truth",
        mask_suffix: str = "_mask",
        image_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    ):
        super().__init__(root=root, name=name)
        self.train_split_dir = train_split_dir
        self.test_split_dir = test_split_dir
        self.train_normal_subdir = train_normal_subdir
        self.test_normal_subdir = test_normal_subdir
        self.ground_truth_dir = ground_truth_dir
        self.mask_suffix = mask_suffix
        self.image_extensions = image_extensions

    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        selected_categories = select_categories(
            available_categories=self.available_categories(),
            requested_categories=categories,
        )
        samples: List[VisionSample] = []

        for category in selected_categories:
            category_root = self.root / category
            train_dir = category_root / self.train_split_dir / self.train_normal_subdir
            test_root = category_root / self.test_split_dir

            train_paths = list_image_files(train_dir, self.image_extensions)
            if max_train_samples_per_category is not None:
                train_paths = train_paths[:max_train_samples_per_category]

            samples.extend(
                [
                    VisionSample(
                        category=category,
                        split="train",
                        image_path=image_path,
                        image_label=0,
                        defect_type=None,
                    )
                    for image_path in train_paths
                ]
            )

            category_test_samples = []
            for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
                image_label = 0 if defect_dir.name == self.test_normal_subdir else 1
                defect_type = None if image_label == 0 else defect_dir.name
                for image_path in list_image_files(defect_dir, self.image_extensions):
                    category_test_samples.append(
                        VisionSample(
                            category=category,
                            split="test",
                            image_path=image_path,
                            image_label=image_label,
                            mask_path=self._resolve_mask_path(category_root, defect_type, image_path),
                            defect_type=defect_type,
                        )
                    )

            if max_test_samples_per_category is not None:
                category_test_samples = category_test_samples[:max_test_samples_per_category]
            samples.extend(category_test_samples)

        return samples

    def _resolve_mask_path(self, category_root: Path, defect_type: Optional[str], image_path: Path) -> Optional[Path]:
        if defect_type is None:
            return None

        mask_dir = category_root / self.ground_truth_dir / defect_type
        if not mask_dir.exists():
            return None

        for suffix in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = mask_dir / f"{image_path.stem}{self.mask_suffix}{suffix}"
            if candidate.exists():
                return candidate
        for suffix in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = mask_dir / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate
        return None


class CsvBenchmarkReader(VisionBenchmarkReader):
    def __init__(
        self,
        root: Union[str, Path],
        name: str,
        csv_path: str,
        category_column: str = "object",
        category_order_column: Optional[str] = None,
        split_column: str = "split",
        train_split_value: str = "train",
        test_split_value: str = "test",
        image_column: str = "image",
        label_column: str = "label",
        normal_label_value: str = "normal",
        mask_column: Optional[str] = "mask",
        defect_type_column: Optional[str] = None,
    ):
        super().__init__(root=root, name=name)
        self.csv_path = csv_path
        self.category_column = category_column
        self.category_order_column = category_order_column
        self.split_column = split_column
        self.train_split_value = train_split_value
        self.test_split_value = test_split_value
        self.image_column = image_column
        self.label_column = label_column
        self.normal_label_value = normal_label_value
        self.mask_column = mask_column
        self.defect_type_column = defect_type_column

    def available_categories(self) -> List[str]:
        rows = self._read_rows()
        if self.category_order_column is not None:
            return self._ordered_categories_from_rows(rows)
        return sorted({row[self.category_column] for row in rows})

    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        rows = self._read_rows()
        if self.category_order_column is not None:
            available_categories = self._ordered_categories_from_rows(rows)
        else:
            available_categories = sorted({row[self.category_column] for row in rows})
        selected_categories = set(
            select_categories(available_categories=available_categories, requested_categories=categories)
        )

        samples_by_key: Dict[Tuple[str, str], List[VisionSample]] = {}
        for row in rows:
            category = row[self.category_column]
            if category not in selected_categories:
                continue

            split = self._resolve_split(row[self.split_column])
            if split is None:
                continue

            image_path = self.root / row[self.image_column]
            if not image_path.exists():
                raise FileNotFoundError(f"Image file referenced in split file does not exist: {image_path}")

            label_raw = row[self.label_column]
            image_label = 0 if label_raw == self.normal_label_value else 1
            mask_path = self._resolve_mask_path(row)
            defect_type = self._resolve_defect_type(row=row, image_label=image_label, label_raw=label_raw)

            sample = VisionSample(
                category=category,
                split=split,
                image_path=image_path,
                image_label=image_label,
                mask_path=mask_path,
                defect_type=defect_type,
            )
            samples_by_key.setdefault((category, split), []).append(sample)

        samples = []
        for category in select_categories(available_categories=available_categories, requested_categories=categories):
            train_samples = samples_by_key.get((category, "train"), [])
            test_samples = samples_by_key.get((category, "test"), [])

            if max_train_samples_per_category is not None:
                train_samples = train_samples[:max_train_samples_per_category]
            if max_test_samples_per_category is not None:
                test_samples = test_samples[:max_test_samples_per_category]

            samples.extend(train_samples)
            samples.extend(test_samples)

        return samples

    def _read_rows(self) -> List[Dict[str, str]]:
        csv_file_path = self.root / self.csv_path
        if not csv_file_path.exists():
            raise FileNotFoundError(f"Benchmark split file not found: {csv_file_path}")

        with csv_file_path.open(newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    def _resolve_split(self, split_value: str) -> Optional[str]:
        if split_value == self.train_split_value:
            return "train"
        if split_value == self.test_split_value:
            return "test"
        return None

    def _resolve_mask_path(self, row: Dict[str, str]) -> Optional[Path]:
        if self.mask_column is None:
            return None

        mask_value = row.get(self.mask_column, "")
        if not mask_value:
            return None

        mask_path = self.root / mask_value
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file referenced in split file does not exist: {mask_path}")
        return mask_path

    def _resolve_defect_type(self, row: Dict[str, str], image_label: int, label_raw: str) -> Optional[str]:
        if image_label == 0:
            return None

        if self.defect_type_column is not None:
            defect_type = row.get(self.defect_type_column, "").strip()
            if defect_type:
                return defect_type

        return label_raw

    def _ordered_categories_from_rows(self, rows: Sequence[Dict[str, str]]) -> List[str]:
        categories_with_order: Dict[str, tuple[int, int]] = {}
        for row_index, row in enumerate(rows):
            category = row[self.category_column]
            raw_order = row.get(self.category_order_column, "").strip()
            order = row_index if not raw_order else self._parse_category_order(raw_order)
            if category in categories_with_order:
                previous_order, previous_row_index = categories_with_order[category]
                if previous_order != order:
                    raise ValueError(
                        f"Category '{category}' has inconsistent values in column "
                        f"'{self.category_order_column}': {previous_order} vs {order}"
                    )
                categories_with_order[category] = (previous_order, min(previous_row_index, row_index))
            else:
                categories_with_order[category] = (order, row_index)

        return [
            category
            for category, _ in sorted(
                categories_with_order.items(),
                key=lambda item: (item[1][0], item[1][1], item[0]),
            )
        ]

    @staticmethod
    def _parse_category_order(raw_value: str) -> int:
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Category order values must be integers, got {raw_value!r}") from exc


class MVTecBenchmarkReader(FolderBenchmarkReader):
    def __init__(self, root: Union[str, Path]):
        super().__init__(root=root, name="mvtec")


class BTechBenchmarkReader(FolderBenchmarkReader):
    def __init__(self, root: Union[str, Path]):
        super().__init__(
            root=root,
            name="btech",
            train_normal_subdir="ok",
            test_normal_subdir="ok",
            ground_truth_dir="ground_truth",
            mask_suffix="",
            image_extensions=(".bmp", ".png", ".jpg", ".jpeg"),
        )


class MPDDBenchmarkReader(FolderBenchmarkReader):
    def __init__(self, root: Union[str, Path]):
        super().__init__(root=root, name="mpdd")


class VisABenchmarkReader(CsvBenchmarkReader):
    def __init__(self, root: Union[str, Path], csv_path: str = "split_csv/1cls.csv"):
        super().__init__(root=root, name="visa", csv_path=csv_path)


class DAGMBenchmarkReader(VisionBenchmarkReader):
    def __init__(self, root: Union[str, Path], include_anomalous_train: bool = False):
        super().__init__(root=root, name="dagm")
        self.include_anomalous_train = include_anomalous_train

    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        selected_categories = select_categories(self.available_categories(), categories)
        samples = []

        for category in selected_categories:
            category_root = self.root / category
            train_samples = self._index_category_split(
                category_root=category_root,
                category=category,
                split_dir_name="Train",
                split_name="train",
                include_anomalies=self.include_anomalous_train,
            )
            test_samples = self._index_category_split(
                category_root=category_root,
                category=category,
                split_dir_name="Test",
                split_name="test",
                include_anomalies=True,
            )

            if max_train_samples_per_category is not None:
                train_samples = train_samples[:max_train_samples_per_category]
            if max_test_samples_per_category is not None:
                test_samples = test_samples[:max_test_samples_per_category]
            samples.extend(train_samples)
            samples.extend(test_samples)

        return samples

    def _index_category_split(
        self,
        category_root: Path,
        category: str,
        split_dir_name: str,
        split_name: str,
        include_anomalies: bool,
    ) -> List[VisionSample]:
        split_dir = category_root / split_dir_name
        label_dir = split_dir / "Label"
        samples = []

        for image_path in list_image_files(split_dir, SUPPORTED_IMAGE_EXTENSIONS):
            mask_path = _dagm_mask_path(label_dir=label_dir, image_path=image_path)
            image_label = 1 if mask_path is not None else 0
            if image_label == 1 and not include_anomalies:
                continue

            samples.append(
                VisionSample(
                    category=category,
                    split=split_name,
                    image_path=image_path,
                    image_label=image_label,
                    mask_path=mask_path,
                    defect_type="defect" if image_label == 1 else None,
                )
            )
        return samples


PREDEFINED_BENCHMARK_READERS: Dict[str, Callable[..., VisionBenchmarkReader]] = {
    "btech": BTechBenchmarkReader,
    "dagm": DAGMBenchmarkReader,
    "mpdd": MPDDBenchmarkReader,
    "mvtec": MVTecBenchmarkReader,
    "visa": VisABenchmarkReader,
}

PREDEFINED_BENCHMARK_ALIASES = {
    "btech_dataset_transformed": "btech",
    "dagm_kaggleupload": "dagm",
    "mvtec_ad": "mvtec",
}


def available_vision_benchmarks() -> List[str]:
    return sorted(PREDEFINED_BENCHMARK_READERS)


def build_vision_benchmark_reader(
    root: Union[str, Path],
    benchmark: Union[str, VisionBenchmarkSpec],
) -> VisionBenchmarkReader:
    root_path = Path(root)
    if isinstance(benchmark, str):
        key = PREDEFINED_BENCHMARK_ALIASES.get(benchmark.lower(), benchmark.lower())
        if key in PREDEFINED_BENCHMARK_READERS:
            return PREDEFINED_BENCHMARK_READERS[key](root=root_path)
        raise ValueError(
            f"Unsupported vision benchmark '{benchmark}'. "
            f"Available presets: {', '.join(available_vision_benchmarks())}"
        )

    if isinstance(benchmark, FolderBenchmarkSpec):
        return FolderBenchmarkReader(root=root_path, **asdict(benchmark))
    return CsvBenchmarkReader(root=root_path, **asdict(benchmark))


def read_vision_benchmark_dataset(
    root: Union[str, Path],
    benchmark: Union[str, VisionBenchmarkSpec],
    dataset_name: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    reader = build_vision_benchmark_reader(root=root, benchmark=benchmark)
    return reader.read_dataset(
        dataset_name=dataset_name,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def index_vision_benchmark(
    root: Union[str, Path],
    benchmark: Union[str, VisionBenchmarkSpec],
    categories: Optional[Sequence[str]] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> List[VisionSample]:
    reader = build_vision_benchmark_reader(root=root, benchmark=benchmark)
    return reader.index_samples(
        categories=categories,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def _dagm_mask_path(label_dir: Path, image_path: Path) -> Optional[Path]:
    if not label_dir.exists():
        return None

    expected_stem = f"{image_path.stem}_label".lower()
    valid_suffixes = {s.lower() for s in SUPPORTED_IMAGE_EXTENSIONS}

    for path in label_dir.iterdir():
        if path.is_file() and path.stem.lower() == expected_stem and path.suffix.lower() in valid_suffixes:
            return path
    return None
