"""General-purpose folder-based vision dataset reader.

Unlike the benchmark-specific readers (MVTec, BTech, VisA, ...),
:class:`GenericFolderReader` handles arbitrary directory layouts commonly
encountered in custom anomaly-detection projects.  Configure the ``layout``
parameter to match your folder structure.

Supported layouts
-----------------

``category/split/label``  *(MVTec-style, default)*
    ``root/<category>/<split>/<label_subdir>/<images>``

``split/category/label``
    ``root/<split>/<category>/<label_subdir>/<images>``

``category/split``
    ``root/<category>/<split>/<images>``
    Train images are assumed normal.  Test images are assumed anomalous.

``split/category``
    ``root/<split>/<category>/<images>``
    Train images are assumed normal.  Test images are assumed anomalous.

``split/label``
    ``root/<split>/<label_subdir>/<images>``
    Single implicit category (name set via ``single_category_name``).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Tuple, Union

from pyclad.data.vision.base import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VisionBenchmarkReader,
    VisionSample,
    list_image_files,
    select_categories,
)


class FolderLayout(str, Enum):
    """Supported directory hierarchy layouts."""

    CATEGORY_SPLIT_LABEL = "category/split/label"
    SPLIT_CATEGORY_LABEL = "split/category/label"
    CATEGORY_SPLIT = "category/split"
    SPLIT_CATEGORY = "split/category"
    SPLIT_LABEL = "split/label"


DEFAULT_NORMAL_LABELS: FrozenSet[str] = frozenset({"good", "ok", "normal"})


def _subdirs(directory: Path) -> List[Path]:
    """Return sorted non-hidden subdirectories of *directory*."""
    if not directory.exists():
        return []
    return sorted(d for d in directory.iterdir() if d.is_dir() and not d.name.startswith("."))


def _is_normal_label(label: str, normal_labels: FrozenSet[str]) -> bool:
    return label.lower() in normal_labels


class GenericFolderReader(VisionBenchmarkReader):
    """Flexible reader for custom vision anomaly-detection datasets.

    Supports multiple directory layouts via the ``layout`` parameter.
    See :class:`FolderLayout` for available options and directory examples.

    Parameters
    ----------
    root:
        Path to the dataset root directory.
    name:
        Dataset name used in the resulting :class:`ConceptsDataset`.
    layout:
        Directory hierarchy.  One of the :class:`FolderLayout` values or an
        equivalent string (e.g. ``"split/category/label"``).
    train_split_dir:
        Name of the training split directory (default ``"train"``).
    test_split_dir:
        Name of the test split directory (default ``"test"``).
    normal_labels:
        Set of subdirectory names treated as *normal* (label 0).  Matching
        is case-insensitive.  Default: ``{"good", "ok", "normal"}``.
    ground_truth_dir:
        Subdirectory containing segmentation masks.  Set to ``None`` to
        disable mask resolution.
    mask_suffix:
        Suffix appended to the image stem when searching for a mask file
        (e.g. ``"_mask"`` → ``image001_mask.png``).
    image_extensions:
        Tuple of accepted image file extensions.
    single_category_name:
        Category name used for layouts without an explicit category level
        (``split/label``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        name: str = "custom",
        layout: Union[str, FolderLayout] = FolderLayout.CATEGORY_SPLIT_LABEL,
        train_split_dir: str = "train",
        test_split_dir: str = "test",
        normal_labels: FrozenSet[str] = DEFAULT_NORMAL_LABELS,
        ground_truth_dir: Optional[str] = "ground_truth",
        mask_suffix: str = "_mask",
        image_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
        single_category_name: str = "default",
    ):
        super().__init__(root=root, name=name)
        self.layout = FolderLayout(layout)
        self.train_split_dir = train_split_dir
        self.test_split_dir = test_split_dir
        self.normal_labels = frozenset(label.lower() for label in normal_labels)
        self.ground_truth_dir = ground_truth_dir
        self.mask_suffix = mask_suffix
        self.image_extensions = image_extensions
        self.single_category_name = single_category_name

    def available_categories(self) -> List[str]:
        if self.layout == FolderLayout.SPLIT_LABEL:
            return [self.single_category_name]

        if self.layout in (
            FolderLayout.CATEGORY_SPLIT_LABEL,
            FolderLayout.CATEGORY_SPLIT,
        ):
            return super().available_categories()

        categories: set[str] = set()
        for split_dir_name in (self.train_split_dir, self.test_split_dir):
            for d in _subdirs(self.root / split_dir_name):
                categories.add(d.name)
        return sorted(categories)

    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        selected: list[str] = select_categories(self.available_categories(), categories)

        if self.layout in (FolderLayout.CATEGORY_SPLIT_LABEL, FolderLayout.SPLIT_CATEGORY_LABEL):
            return self._index_labeled(selected, max_train_samples_per_category, max_test_samples_per_category)
        if self.layout in (FolderLayout.CATEGORY_SPLIT, FolderLayout.SPLIT_CATEGORY):
            return self._index_flat(selected, max_train_samples_per_category, max_test_samples_per_category)
        if self.layout == FolderLayout.SPLIT_LABEL:
            return self._index_split_label(selected, max_train_samples_per_category, max_test_samples_per_category)

        raise ValueError(f"Unsupported layout: {self.layout}")

    def _resolve_split_root(self, category: str, split_dir: str) -> Path:
        if self.layout in (FolderLayout.CATEGORY_SPLIT_LABEL, FolderLayout.CATEGORY_SPLIT):
            return self.root / category / split_dir
        return self.root / split_dir / category

    def _resolve_mask_root(self, category: str) -> Path:
        if self.layout == FolderLayout.CATEGORY_SPLIT_LABEL:
            return self.root / category
        return self._resolve_split_root(category, self.test_split_dir)

    def _index_labeled(
        self,
        categories: Sequence[str],
        max_train: Optional[int],
        max_test: Optional[int],
    ) -> List[VisionSample]:
        samples: List[VisionSample] = []
        for category in categories:
            train_root = self._resolve_split_root(category, self.train_split_dir)
            train_samples: List[VisionSample] = []
            for label_dir in _subdirs(train_root):
                if not _is_normal_label(label_dir.name, self.normal_labels):
                    continue
                for img in list_image_files(label_dir, self.image_extensions):
                    train_samples.append(VisionSample(category=category, split="train", image_path=img, image_label=0))
            if max_train is not None:
                train_samples = train_samples[:max_train]
            samples.extend(train_samples)

            test_root = self._resolve_split_root(category, self.test_split_dir)
            mask_root = self._resolve_mask_root(category)
            test_samples: List[VisionSample] = []
            for label_dir in _subdirs(test_root):
                is_normal = _is_normal_label(label_dir.name, self.normal_labels)
                image_label = 0 if is_normal else 1
                defect_type = None if is_normal else label_dir.name
                for img in list_image_files(label_dir, self.image_extensions):
                    test_samples.append(
                        VisionSample(
                            category=category,
                            split="test",
                            image_path=img,
                            image_label=image_label,
                            mask_path=self._find_mask(mask_root, defect_type, img),
                            defect_type=defect_type,
                        )
                    )
            if max_test is not None:
                test_samples = test_samples[:max_test]
            samples.extend(test_samples)

        return samples

    def _index_flat(
        self,
        categories: Sequence[str],
        max_train: Optional[int],
        max_test: Optional[int],
    ) -> List[VisionSample]:
        samples: List[VisionSample] = []
        for category in categories:
            train_dir = self._resolve_split_root(category, self.train_split_dir)
            train_samples = [
                VisionSample(category=category, split="train", image_path=img, image_label=0)
                for img in list_image_files(train_dir, self.image_extensions)
            ]
            if max_train is not None:
                train_samples = train_samples[:max_train]
            samples.extend(train_samples)

            test_dir = self._resolve_split_root(category, self.test_split_dir)
            test_samples = [
                VisionSample(category=category, split="test", image_path=img, image_label=1)
                for img in list_image_files(test_dir, self.image_extensions)
            ]
            if max_test is not None:
                test_samples = test_samples[:max_test]
            samples.extend(test_samples)

        return samples

    def _index_split_label(
        self,
        categories: Sequence[str],
        max_train: Optional[int],
        max_test: Optional[int],
    ) -> List[VisionSample]:
        category = self.single_category_name
        samples: List[VisionSample] = []

        train_root = self.root / self.train_split_dir
        train_samples: List[VisionSample] = []
        for label_dir in _subdirs(train_root):
            if not _is_normal_label(label_dir.name, self.normal_labels):
                continue
            for img in list_image_files(label_dir, self.image_extensions):
                train_samples.append(VisionSample(category=category, split="train", image_path=img, image_label=0))
        if max_train is not None:
            train_samples = train_samples[:max_train]
        samples.extend(train_samples)

        test_root = self.root / self.test_split_dir
        test_samples: List[VisionSample] = []
        for label_dir in _subdirs(test_root):
            is_normal = _is_normal_label(label_dir.name, self.normal_labels)
            image_label = 0 if is_normal else 1
            defect_type = None if is_normal else label_dir.name
            for img in list_image_files(label_dir, self.image_extensions):
                test_samples.append(
                    VisionSample(
                        category=category,
                        split="test",
                        image_path=img,
                        image_label=image_label,
                        defect_type=defect_type,
                    )
                )
        if max_test is not None:
            test_samples = test_samples[:max_test]
        samples.extend(test_samples)

        return samples

    def _find_mask(
        self,
        context_root: Path,
        defect_type: Optional[str],
        image_path: Path,
    ) -> Optional[Path]:
        """Search for a segmentation mask matching *image_path*.

        Looks in ``<context_root>/<ground_truth_dir>/<defect_type>/`` for a
        file whose stem matches ``<image_stem><mask_suffix>`` or just
        ``<image_stem>``.  Returns ``None`` when no mask is found or when
        mask resolution is disabled (``ground_truth_dir is None``).
        """
        if self.ground_truth_dir is None or defect_type is None:
            return None

        mask_dir = context_root / self.ground_truth_dir / defect_type
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

    @staticmethod
    def detect_layout(root: Union[str, Path]) -> FolderLayout:
        """Heuristically detect the directory layout under *root*.

        Inspects the first two levels of subdirectories.  If the top-level
        directories look like split names (``train`` / ``test``), the layout
        starts with ``split/…``; otherwise it starts with ``category/…``.
        A third level of subdirectories indicates the ``…/label`` variant.

        Raises :class:`ValueError` if the layout cannot be determined.
        """
        root_path = Path(root)
        split_names = {"train", "test", "val", "validation"}

        top_dirs = _subdirs(root_path)
        if not top_dirs:
            raise ValueError(f"Cannot detect layout: no subdirectories in {root_path}")

        top_names = {d.name.lower() for d in top_dirs}
        top_looks_like_splits = bool(top_names & split_names)

        if top_looks_like_splits:
            # root/train/... or root/test/...
            # Look one level deeper to decide between split/category and
            # split/category/label (or split/label).
            sample_split_dir = top_dirs[0]
            second_level = _subdirs(sample_split_dir)
            if not second_level:
                raise ValueError(f"Cannot detect layout: no subdirectories in {sample_split_dir}")

            label_like_names = {"good", "ok", "normal", "anomaly", "defect", "abnormal"}
            second_names = {d.name.lower() for d in second_level}
            if second_names & label_like_names:
                return FolderLayout.SPLIT_LABEL

            third_level = _subdirs(second_level[0])
            if third_level:
                return FolderLayout.SPLIT_CATEGORY_LABEL

            return FolderLayout.SPLIT_CATEGORY

        first_category = top_dirs[0]
        second_level = _subdirs(first_category)
        if not second_level:
            raise ValueError(f"Cannot detect layout: no subdirectories in {first_category}")

        second_names = {d.name.lower() for d in second_level}
        if not (second_names & split_names):
            raise ValueError(
                f"Cannot detect layout: expected split directories (train/test) "
                f"in {first_category}, found: {sorted(d.name for d in second_level)}"
            )
        sample_split = next((d for d in second_level if d.name.lower() in split_names), second_level[0])
        third_level = _subdirs(sample_split)
        if third_level:
            return FolderLayout.CATEGORY_SPLIT_LABEL
        return FolderLayout.CATEGORY_SPLIT
