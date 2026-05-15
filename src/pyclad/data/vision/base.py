from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.vision._utils import resolve_category_order
from pyclad.data.vision_concept import VisionConcept

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class VisionSample:
    category: str
    split: str
    image_path: Path
    image_label: int
    mask_path: Optional[Path] = None
    defect_type: Optional[str] = None


class VisionBenchmarkReader(ABC):
    def __init__(self, root: Union[str, Path], name: str):
        self.root = Path(root)
        self.name = name

    def available_categories(self) -> List[str]:
        """List categories by scanning top-level subdirectories of root.

        Subclasses may override for layouts where categories are not
        direct children of root (e.g. CSV-driven benchmarks).
        """
        return sorted(
            category_dir.name
            for category_dir in self.root.iterdir()
            if category_dir.is_dir() and not category_dir.name.startswith(".")
        )

    @abstractmethod
    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        raise NotImplementedError

    def read_dataset(
        self,
        dataset_name: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        data_mode: str = "numpy",
        resize_to: Optional[Tuple[int, int]] = None,
        color_mode: str = "rgb",
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> ConceptsDataset:
        validate_read_options(data_mode=data_mode, color_mode=color_mode)
        samples = self.index_samples(
            categories=categories,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
        )
        return build_concepts_dataset_from_samples(
            samples=samples,
            categories=categories,
            dataset_name=dataset_name or f"{self.name.upper()}-VisionBenchmark",
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
        )


def build_concepts_dataset_from_samples(
    samples: Sequence[VisionSample],
    dataset_name: str,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
) -> ConceptsDataset:
    """Build a ConceptsDataset from a flat list of indexed VisionSamples.

    Iterates over categories (in the order given by *categories*, or inferred
    from the first-occurrence order in *samples*), splits each category into
    train and test subsets, materializes image data, and returns a ready-to-use
    :class:`ConceptsDataset`.
    """
    selected_categories = resolve_category_order(samples=samples, categories=categories)

    train_concepts: List[Concept] = []
    test_concepts: List[Concept] = []

    for category in selected_categories:
        train_samples = [s for s in samples if s.category == category and s.split == "train"]
        test_samples = [s for s in samples if s.category == category and s.split == "test"]

        train_concepts.append(
            Concept(
                name=category,
                data=materialize_samples(
                    samples=train_samples,
                    data_mode=data_mode,
                    resize_to=resize_to,
                    color_mode=color_mode,
                ),
                labels=None,
            )
        )

        if len(test_samples) > 0:
            has_any_mask = any(s.mask_path is not None for s in test_samples)
            if has_any_mask:
                from pyclad.data.vision.masks import load_ground_truth_masks_for_samples

                masks, kept_indices = load_ground_truth_masks_for_samples(
                    test_samples,
                    resize_to=resize_to,
                )
                test_concepts.append(
                    VisionConcept(
                        name=category,
                        data=materialize_samples(
                            samples=test_samples,
                            data_mode=data_mode,
                            resize_to=resize_to,
                            color_mode=color_mode,
                        ),
                        labels=np.asarray([s.image_label for s in test_samples], dtype=np.int64),
                        masks=masks,
                    )
                )
            else:
                test_concepts.append(
                    Concept(
                        name=category,
                        data=materialize_samples(
                            samples=test_samples,
                            data_mode=data_mode,
                            resize_to=resize_to,
                            color_mode=color_mode,
                        ),
                        labels=np.asarray([s.image_label for s in test_samples], dtype=np.int64),
                    )
                )

    return ConceptsDataset(
        name=dataset_name,
        train_concepts=train_concepts,
        test_concepts=test_concepts,
    )


def validate_read_options(data_mode: str, color_mode: str) -> None:
    if data_mode not in {"numpy", "paths"}:
        raise ValueError("data_mode must be one of: 'numpy', 'paths'")
    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError("color_mode must be one of: 'rgb', 'grayscale'")


def select_categories(
    available_categories: Sequence[str],
    requested_categories: Optional[Sequence[str]] = None,
) -> List[str]:
    if requested_categories is None:
        return list(available_categories)

    missing = sorted(set(requested_categories) - set(available_categories))
    if missing:
        raise ValueError(
            f"Requested categories not found: {missing}. Available categories: {list(available_categories)}"
        )
    return list(requested_categories)


def list_image_files(directory: Path, image_extensions: Iterable[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    suffixes = {extension.lower() for extension in image_extensions}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def materialize_samples(
    samples: Sequence[VisionSample],
    data_mode: str,
    resize_to: Optional[Tuple[int, int]],
    color_mode: str,
) -> np.ndarray:
    if data_mode == "paths":
        return np.asarray([str(sample.image_path) for sample in samples], dtype=object)

    arrays = [_load_image(sample.image_path, resize_to=resize_to, color_mode=color_mode) for sample in samples]
    if len(arrays) == 0:
        return np.asarray([], dtype=np.float32)

    try:
        return np.stack(arrays, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Could not stack image arrays into a single batch. "
            "Provide resize_to=(height, width) so every image materializes to the same shape."
        ) from exc


def _load_image(image_path: Path, resize_to: Optional[Tuple[int, int]], color_mode: str) -> np.ndarray:
    resampling_enum = getattr(Image, "Resampling", Image)
    target_mode = "RGB" if color_mode == "rgb" else "L"

    with Image.open(image_path) as image:
        image = image.convert(target_mode)
        if resize_to is not None:
            image = image.resize((resize_to[1], resize_to[0]), resampling_enum.BILINEAR)
        array = np.asarray(image)

    if color_mode == "grayscale":
        array = array[..., None]
    return array
