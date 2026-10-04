from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data._utils import resolve_category_order
from pyclad.vision.data.masks import load_ground_truth_masks_for_samples
from pyclad.vision.data.parallel import map_in_threads
from pyclad.vision.data.sample import VisionSample
from pyclad.vision.data.vision_concept import VisionConcept

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DATA_MODES = ("numpy", "paths")
COLOR_MODES = {"rgb": "RGB", "grayscale": "L"}
INTERPOLATIONS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


@dataclass(frozen=True)
class ImageLoadOptions:
    data_mode: str = "numpy"
    resize_to: Optional[Tuple[int, int]] = None
    color_mode: str = "rgb"
    interpolation: str = "bilinear"
    apply_exif_transpose: bool = False

    def __post_init__(self) -> None:
        if self.data_mode not in DATA_MODES:
            raise ValueError(f"data_mode must be one of: {DATA_MODES}, got {self.data_mode!r}")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of: {tuple(COLOR_MODES)}, got {self.color_mode!r}")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of: {tuple(INTERPOLATIONS)}, got {self.interpolation!r}")


class VisionBenchmarkReader(ABC):
    def __init__(self, root: Union[str, Path], name: str):
        self.root = Path(root) if root is not None else None
        self.name = name

    def available_categories(self) -> List[str]:
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
        interpolation: str = "bilinear",
        apply_exif_transpose: bool = False,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> ConceptsDataset:
        options = ImageLoadOptions(
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            interpolation=interpolation,
            apply_exif_transpose=apply_exif_transpose,
        )
        samples = self.index_samples(
            categories=categories,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
        )
        return build_concepts_dataset_from_samples(
            samples=samples,
            categories=categories,
            dataset_name=dataset_name or f"{self.name.upper()}-VisionBenchmark",
            options=options,
        )


def build_vision_concept(
    name: str,
    samples: Sequence[VisionSample],
    options: ImageLoadOptions,
    with_labels: bool,
) -> Concept:
    if not with_labels:
        return Concept(name=name, data=materialize_samples(samples, options), labels=None)

    if not any(sample.mask_path is not None for sample in samples):
        return Concept(
            name=name,
            data=materialize_samples(samples, options),
            labels=np.asarray([sample.image_label for sample in samples], dtype=np.int64),
        )

    masks, kept_indices = load_ground_truth_masks_for_samples(samples, resize_to=options.resize_to)
    kept_samples = [samples[index] for index in kept_indices]
    return VisionConcept(
        name=name,
        data=materialize_samples(kept_samples, options),
        labels=np.asarray([sample.image_label for sample in kept_samples], dtype=np.int64),
        masks=masks,
    )


def build_concepts_dataset_from_samples(
    samples: Sequence[VisionSample],
    dataset_name: str,
    categories: Optional[Sequence[str]] = None,
    options: Optional[ImageLoadOptions] = None,
) -> ConceptsDataset:
    options = options if options is not None else ImageLoadOptions()
    selected_categories = resolve_category_order(samples=samples, categories=categories)

    buckets: Dict[Tuple[str, str], List[VisionSample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.category, sample.split)].append(sample)

    train_concepts: List[Concept] = []
    test_concepts: List[Concept] = []
    for category in selected_categories:
        train_concepts.append(
            build_vision_concept(
                name=category,
                samples=buckets.get((category, "train"), []),
                options=options,
                with_labels=False,
            )
        )
        test_samples = buckets.get((category, "test"), [])
        if test_samples:
            test_concepts.append(
                build_vision_concept(name=category, samples=test_samples, options=options, with_labels=True)
            )

    return ConceptsDataset(name=dataset_name, train_concepts=train_concepts, test_concepts=test_concepts)


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
    candidates = directory.iterdir()
    return sorted(path for path in candidates if path.is_file() and path.suffix.lower() in suffixes)


def materialize_samples(samples: Sequence[VisionSample], options: ImageLoadOptions) -> np.ndarray:
    if options.data_mode == "paths":
        return np.asarray([str(sample.image_path) for sample in samples], dtype=object)

    arrays = map_in_threads(partial(_load_image, options=options), [sample.image_path for sample in samples])
    if len(arrays) == 0:
        return np.asarray([], dtype=np.float32)

    try:
        return np.stack(arrays, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Could not stack image arrays into a single batch. "
            "Provide resize_to=(height, width) so every image materializes to the same shape."
        ) from exc


def _load_image(image_path: Path, options: ImageLoadOptions) -> np.ndarray:
    with Image.open(image_path) as image:
        if options.apply_exif_transpose:
            image = ImageOps.exif_transpose(image)
        image = image.convert(COLOR_MODES[options.color_mode])
        if options.resize_to is not None:
            image = image.resize(
                (options.resize_to[1], options.resize_to[0]),
                INTERPOLATIONS[options.interpolation],
            )
        array = np.asarray(image)

    if options.color_mode == "grayscale":
        array = array[..., None]
    return array
