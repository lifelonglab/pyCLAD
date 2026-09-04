from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from pyclad.vision.models.ucad.config import UCADConfig

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_MULTIDATASET_SEPARATOR = "__"


def category_of(concept_id: str) -> str:
    """Category part of a concept id, dropping a multidataset ``<alias>__`` prefix.

    Deliberately duplicated from ReplayCAD's helper rather than imported: a model must not
    depend on a strategy package.
    """
    return str(concept_id).rsplit(_MULTIDATASET_SEPARATOR, 1)[-1]


class StructureMaskProvider(abc.ABC):
    """Supplies SAM region-id maps for one task's training images."""

    @abc.abstractmethod
    def masks_for(self, concept_id: Optional[str], images: np.ndarray) -> Optional[np.ndarray]:
        """Return ``(N, H, W)`` int64 region labels aligned with ``images``, or ``None``."""


class NoStructureMaskProvider(StructureMaskProvider):
    """CPM-only ablation: no masks, therefore no SCL and an untrained prompt."""

    def masks_for(self, concept_id: Optional[str], images: np.ndarray) -> None:
        return None


class PrecomputedStructureMaskProvider(StructureMaskProvider):
    """Loads the authors' precomputed SAM maps (the ``mvtec2d-sam-b`` archive).

    Region ids live in channel 0 of the PNG, read the same way as the reference
    implementation's ``cv2.imread(path)[:, :, 0]`` (see ``_load_label_map``). Masks are paired
    with images by sorted filename order — the same order the vision readers index a category
    in — and a count mismatch raises rather than silently shifting the pairing by one.
    """

    def __init__(self, root: Union[str, Path], interpolation: str = "bilinear"):
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError(f"interpolation must be 'bilinear' or 'nearest', got {interpolation!r}")
        self.root = Path(root).expanduser()
        self.interpolation = interpolation

    def masks_for(self, concept_id: Optional[str], images: np.ndarray) -> np.ndarray:
        directory = self._resolve_directory(concept_id)
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        if len(paths) != len(images):
            raise ValueError(
                f"Concept '{concept_id}' has {len(images)} training image(s) but "
                f"{len(paths)} structure mask(s) in {directory}. Masks are paired by sorted "
                "filename order, so the counts must match exactly."
            )

        height, width = int(images.shape[1]), int(images.shape[2])
        return np.stack([self._load_label_map(path, height, width) for path in paths]).astype(np.int64, copy=False)

    def _resolve_directory(self, concept_id: Optional[str]) -> Path:
        if not concept_id:
            raise ValueError(
                "UCAD structure_mode='precomputed' needs a concept id to locate that concept's "
                f"structure masks, but none was supplied (concept_id={concept_id!r}). This means "
                "fit() ran without a concept-aware scenario: use a scenario such as "
                "ConceptAwareScenario, whose strategy calls UCAD.set_current_concept() before "
                "fit(), so precomputed masks can be paired with the right task."
            )
        candidates: list[Path] = []
        names = [str(concept_id)]
        category = category_of(concept_id)
        if category != str(concept_id):
            names.append(category)
        for name in names:
            candidates += [self.root / name / "train" / "good", self.root / name / "train", self.root / name]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            "Could not locate precomputed UCAD structure masks. Checked: " + ", ".join(str(path) for path in candidates)
        )

    def _load_label_map(self, path: Path, height: int, width: int) -> np.ndarray:
        with Image.open(path) as image:
            label_map = image.getchannel(0)
            if label_map.size != (width, height):
                resample = Image.Resampling.BILINEAR if self.interpolation == "bilinear" else Image.Resampling.NEAREST
                label_map = label_map.resize((width, height), resample=resample)
            return np.asarray(label_map, dtype=np.int64)


class SamStructureMaskProvider(StructureMaskProvider):
    """Generates multi-region SAM maps at run time.

    Regions are painted largest-first so that smaller regions overwrite larger ones and every
    region ends up with a distinct id, matching the authors' preprocessing script.
    """

    def __init__(self, checkpoint: Union[str, Path], model_type: str, device: str):
        self.checkpoint = Path(checkpoint).expanduser()
        self.model_type = model_type
        self.device = device
        self._generator = None

    def _load(self):
        if self._generator is not None:
            return self._generator
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "UCAD's online SAM structure masks require segment-anything. " "Install pyclad with the 'ucad' extra."
            ) from exc
        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        sam.to(device=self.device)
        sam.eval()
        self._generator = SamAutomaticMaskGenerator(sam)
        return self._generator

    def masks_for(self, concept_id: Optional[str], images: np.ndarray) -> np.ndarray:
        generator = self._load()
        return np.stack([self._one(generator, image) for image in images]).astype(np.int64, copy=False)

    @staticmethod
    def _one(generator, image: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.dtype != np.uint8:
            scale = 255.0 if rgb.size and float(rgb.max()) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb[..., :3])

        labels = np.zeros(rgb.shape[:2], dtype=np.int64)
        for region_id, region in enumerate(sorted(generator.generate(rgb), key=lambda r: r["area"], reverse=True), 1):
            labels[np.asarray(region["segmentation"], dtype=bool)] = region_id
        return labels


def build_structure_mask_provider(config: UCADConfig, device: str) -> StructureMaskProvider:
    if config.structure_mode == "none":
        return NoStructureMaskProvider()
    if config.structure_mode == "precomputed":
        return PrecomputedStructureMaskProvider(config.structure_mask_root, config.mask_interpolation)
    if config.structure_mode == "sam":
        return SamStructureMaskProvider(config.sam_checkpoint, config.sam_model_type, device)
    raise ValueError(f"Unknown UCAD structure mode: {config.structure_mode}")
