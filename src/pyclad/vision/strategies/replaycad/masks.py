from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from scipy import ndimage

from pyclad.vision.strategies.replaycad.config import ReplayCADConfig, TrainAugmentation

_MULTIDATASET_SEPARATOR = "__"


def category_of(concept_id: str) -> str:
    """Category part of a concept id, dropping a multidataset ``<alias>__`` prefix."""
    return concept_id.rsplit(_MULTIDATASET_SEPARATOR, 1)[-1]


class MaskProvider(Protocol):
    def masks_for(self, concept_id: str, images: np.ndarray) -> np.ndarray:
        """Return ``(N, H, W)`` uint8 masks aligned with ``images`` on the batch dimension."""


class FullFrameMaskProvider:
    """All-ones masks, for texture concepts where SAM returns no object mask."""

    def masks_for(self, concept_id: str, images: np.ndarray) -> np.ndarray:
        height, width = images.shape[1], images.shape[2]
        return np.full((len(images), height, width), 255, dtype=np.uint8)


class PrecomputedMaskProvider:
    """Loads the ReplayCAD authors' SAM masks from ``<root>/<benchmark>/<category>/*.png``."""

    def __init__(self, root: Path, benchmark: str):
        self.root = Path(root)
        self.benchmark = benchmark

    def masks_for(self, concept_id: str, images: np.ndarray) -> np.ndarray:
        directory = self.root / self.benchmark / category_of(concept_id)
        if not directory.is_dir():
            raise ValueError(f"No precomputed masks for concept '{concept_id}' at {directory}")

        paths = sorted(directory.glob("*.png"))
        if len(paths) != len(images):
            raise ValueError(
                f"Concept '{concept_id}' has {len(images)} training images but "
                f"{len(paths)} precomputed mask(s) in {directory}. Masks are matched by sorted "
                "filename order, so the counts must be equal."
            )

        height, width = images.shape[1], images.shape[2]
        masks = []
        for path in paths:
            mask = Image.open(path).convert("L")
            if mask.size != (width, height):
                mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
            masks.append(np.where(np.asarray(mask, dtype=np.uint8) >= 128, 255, 0).astype(np.uint8))
        return np.stack(masks)


class SamMaskProvider:
    """Computes object masks with Segment Anything, keeping the largest returned region."""

    def __init__(self, checkpoint: Path, model_type: str, device: str):
        self.checkpoint = Path(checkpoint)
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
                "ReplayCAD's SAM mask backend requires segment-anything. " "Install pyclad with the 'replaycad' extra."
            ) from exc

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        sam.to(self.device)
        self._generator = SamAutomaticMaskGenerator(sam)
        return self._generator

    def masks_for(self, concept_id: str, images: np.ndarray) -> np.ndarray:
        generator = self._load()
        height, width = images.shape[1], images.shape[2]

        masks = []
        for image in images:
            rgb = np.asarray(image, dtype=np.uint8)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=-1)
            regions = generator.generate(rgb[..., :3])
            if not regions:
                masks.append(np.full((height, width), 255, dtype=np.uint8))
                continue
            largest = max(regions, key=lambda region: int(region["area"]))
            masks.append((np.asarray(largest["segmentation"], dtype=bool) * 255).astype(np.uint8))
        return np.stack(masks)


class MaskModeRouter:
    """Sends concepts listed as ``full-frame`` in ``mask_modes`` to the full-frame provider."""

    def __init__(self, base: MaskProvider, full_frame_categories: frozenset[str]):
        self.base = base
        self.full_frame = FullFrameMaskProvider()
        self.full_frame_categories = full_frame_categories

    def masks_for(self, concept_id: str, images: np.ndarray) -> np.ndarray:
        if category_of(concept_id) in self.full_frame_categories:
            return self.full_frame.masks_for(concept_id, images)
        return self.base.masks_for(concept_id, images)


def build_mask_provider(config: ReplayCADConfig, benchmark: str) -> MaskProvider:
    if config.mask_backend == "full-frame":
        return FullFrameMaskProvider()
    if config.mask_backend == "precomputed":
        base: MaskProvider = PrecomputedMaskProvider(root=config.precomputed_mask_root, benchmark=benchmark)
    else:
        base = SamMaskProvider(checkpoint=config.sam_checkpoint, model_type=config.sam_model_type, device=config.device)

    full_frame = frozenset(name for name, mode in config.mask_modes.items() if mode == "full-frame")
    return MaskModeRouter(base, full_frame) if full_frame else base


def augment_mask(mask: np.ndarray, config: ReplayCADConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply this concept's generation-time mask transform, dispatched on ``config.mask_augmentation``.

    ``"paper"`` is the paper's own prose description ("randomly rotate and shift the stored
    masks"), distinct from the five named transforms below, which reproduce the release's
    ``mask_transfor.py``; see ``per_class.py`` for which class uses which. ``"none"`` is the
    default and covers the rest.
    """
    mode = config.mask_augmentation
    if mode == "none":
        return mask
    elif mode == "random_rotate":
        return random_rotate(mask, rng)
    elif mode == "random_3directions_rotate":
        return random_3directions_rotate(mask, rng)
    elif mode == "little_rotate_and_move":
        return little_rotate_and_move(
            mask,
            rng,
            angles=config.mask_transform_angles,
            distance=config.mask_transform_distance,
            transpose=config.mask_transform_transpose,
        )
    elif mode == "visa_candle":
        return visa_candle(mask, rng, shift_pixels=config.visa_candle_shift_pixels, rotate=config.visa_candle_rotate)
    elif mode == "random_reset":
        return random_reset(mask, rng)
    elif mode != "paper":
        raise ValueError(f"Unknown mask_augmentation mode '{mode}'")

    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    if config.max_mask_rotation_degrees > 0:
        angle = float(rng.uniform(0.0, config.max_mask_rotation_degrees))
        image = image.rotate(angle, expand=False, fillcolor=0)

    shifted = np.asarray(image, dtype=np.uint8)
    max_shift_y = int(round(shifted.shape[0] * config.max_mask_shift_fraction))
    max_shift_x = int(round(shifted.shape[1] * config.max_mask_shift_fraction))
    if max(max_shift_y, max_shift_x) > 0:
        dy = int(rng.integers(-max_shift_y, max_shift_y + 1)) if max_shift_y else 0
        dx = int(rng.integers(-max_shift_x, max_shift_x + 1)) if max_shift_x else 0
        shifted = _translate(shifted, dy, dx)

    return _binarize(shifted)


def random_rotate(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Whole-mask rotation, uniform in [0, 360] degrees, no shift. Mirrors the release's
    ``random_rorate``, the sole generation-time transform for metal_nut, screw, hazelnut and fryum.
    """
    angle = int(rng.integers(0, 361))
    return _rotate_mask(mask, angle)


def random_3directions_rotate(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One of six equally likely outcomes: rotate 90/180/270 degrees, flip either axis, or leave
    unchanged. Mirrors ``random_3direcions_rotate``, the authors' sole transform for grid --
    "unchanged" is a real one-in-six outcome, not a fallback for an unhandled case.
    """
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    choice = int(rng.integers(0, 6))
    if choice == 0:
        image = image.rotate(90)
    elif choice == 1:
        image = image.rotate(180)
    elif choice == 2:
        image = image.rotate(270)
    elif choice == 3:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif choice == 4:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # choice == 5: identity.
    return _binarize(np.asarray(image, dtype=np.uint8))


def little_rotate_and_move(
    mask: np.ndarray,
    rng: np.random.Generator,
    angles: int = 2,
    distance: float = 0.05,
    transpose: bool = False,
) -> np.ndarray:
    """A small-angle rotation plus a small shift; mirrors ``little_rorate_and_move``.

    The authors' transform for transistor (defaults), cashew (angles=10), chewinggum and
    pcb1-pcb3 (distance=0.1 or 0.02, transpose=True), and pcb4 (distance=0.02).
    """
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))

    if transpose and int(rng.integers(0, 2)) == 1:
        image = image.rotate(180)

    angle = int(rng.integers(0, angles + 1))
    angle = int(rng.choice([angle, 360 - angle]))
    image = image.rotate(angle, expand=False, fillcolor=0)

    shifted = np.asarray(image, dtype=np.uint8)
    height, width = shifted.shape[:2]
    upper_bound = int(width * distance)
    magnitude = min(int(rng.integers(0, upper_bound + 1)), height // 2, width // 2)

    direction = int(rng.integers(0, 4))
    if direction == 0:
        shifted = _translate(shifted, magnitude, 0)
    elif direction == 1:
        shifted = _translate(shifted, -magnitude, 0)
    elif direction == 2:
        shifted = _translate(shifted, 0, magnitude)
    else:
        shifted = _translate(shifted, 0, -magnitude)

    return _binarize(shifted)


def visa_candle(
    mask: np.ndarray,
    rng: np.random.Generator,
    shift_pixels: int = 20,
    rotate: bool = False,
) -> np.ndarray:
    """Detach one random connected component, translate it, optionally rotate it, then remerge.

    Mirrors ``visa_candle``; the authors' transform for candle (defaults) and macaroni1
    (rotate=True). Uses ``scipy.ndimage.label`` (4-connectivity) rather than the original's
    ``cv2.findContours(RETR_EXTERNAL)`` (8-connectivity, filled contours) -- equivalent for the
    hole-free, non-diagonally-touching components the affected VisA masks actually have.
    """
    array = np.asarray(mask, dtype=np.uint8)
    labeled, num_features = ndimage.label(array >= 128)
    if num_features == 0:
        return _binarize(array)

    chosen = int(rng.integers(1, num_features + 1))
    component = labeled == chosen
    remainder = np.where(component, 0, array).astype(np.uint8)
    piece = np.where(component, 255, 0).astype(np.uint8)

    dy = int(rng.integers(-shift_pixels, shift_pixels + 1))
    dx = int(rng.integers(-shift_pixels, shift_pixels + 1))
    piece = _translate(piece, dy, dx)

    if rotate:
        angle = float(rng.uniform(-10.0, 10.0))
        piece = _rotate_mask(piece, angle)

    return _binarize(np.maximum(remainder, piece))


def random_reset(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crop each connected component to its bounding box and re-place it at a random,
    non-overlapping position on an otherwise black canvas. Mirrors ``radomreset``; not reachable
    by any released class (see ``per_class.py``). A component that exhausts 100 placement
    attempts is dropped rather than misplaced (see docs/vision.md's divergences list).
    """
    array = np.asarray(mask, dtype=np.uint8)
    height, width = array.shape[:2]
    labeled, num_features = ndimage.label(array >= 128)

    canvas = np.zeros_like(array)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for component_id in range(1, num_features + 1):
        rows, cols = np.nonzero(labeled == component_id)
        y0, x0, y1, x1 = rows.min(), cols.min(), rows.max(), cols.max()
        patch = array[y0 : y1 + 1, x0 : x1 + 1]
        patch_height, patch_width = patch.shape

        for _ in range(100):
            x = int(rng.integers(0, width - patch_width + 1))
            y = int(rng.integers(0, height - patch_height + 1))
            box = (x, y, x + patch_width, y + patch_height)
            if not any(_boxes_overlap(box, other) for other in placed_boxes):
                placed_boxes.append(box)
                canvas[y : y + patch_height, x : x + patch_width] = patch
                break

    return _binarize(canvas)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    rotated = image.rotate(angle, expand=False, fillcolor=0)
    return _binarize(np.asarray(rotated, dtype=np.uint8))


def _translate(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift by ``(dy, dx)`` pixels: the vacated edge is filled with black rather than wrapping
    (``np.roll``'s default), and content pushed past the far edge is discarded.
    """
    shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted


def _binarize(array: np.ndarray) -> np.ndarray:
    """Snap back to the {0, 255} contract after an operation (typically rotation) that can leave
    interpolated, non-binary values.
    """
    return np.where(np.asarray(array) >= 128, 255, 0).astype(np.uint8)


def _border_mean_colour(image: np.ndarray, strip: int = 20) -> tuple[int, int, int]:
    """Mean colour of four border strips, used to fill the corners a rotation leaves empty."""
    height, width = image.shape[0], image.shape[1]
    strips = [
        image[max(height - strip, 0) :, :, :],
        image[:, : min(strip, width), :],
        image[: min(strip, height), :, :],
        image[:, max(width - strip, 0) :, :],
    ]
    pixels = np.concatenate([piece.reshape(-1, image.shape[2]) for piece in strips], axis=0)
    return tuple(int(value) for value in pixels.mean(axis=0))


def augment_training_pair(
    image: np.ndarray, mask: np.ndarray, mode: TrainAugmentation, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Augment one image/mask pair identically during compression (mirrors
    ``personalized.py:214-280``); independent transforms would leave the mask no longer
    describing its paired image.
    """
    if mode == "none":
        return image, mask

    image_pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    mask_pil = Image.fromarray(np.asarray(mask, dtype=np.uint8))

    if mode == "rotate_180":
        if int(rng.integers(0, 2)) == 0:
            image_pil = image_pil.rotate(180)
            mask_pil = mask_pil.rotate(180)
    elif mode == "rotate_3_directions":
        choice = int(rng.integers(0, 6))
        if choice in (0, 1, 2):
            angle = (choice + 1) * 90
            image_pil = image_pil.rotate(angle)
            mask_pil = mask_pil.rotate(angle)
        elif choice == 3:
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif choice == 4:
            image_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
            mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
    elif mode == "rotate_all_directions":
        angle = int(rng.integers(0, 361))
        image_pil = image_pil.rotate(angle, expand=False, fillcolor=_border_mean_colour(np.asarray(image)))
        mask_pil = mask_pil.rotate(angle, expand=False, fillcolor=0)
    else:
        raise ValueError(f"Unknown train_augmentation mode '{mode}'")

    return np.asarray(image_pil, dtype=np.uint8), np.asarray(mask_pil, dtype=np.uint8)


def select_stored_masks(masks: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick up to ``count`` masks to store in the artifact (the paper's M in [1, 10])."""
    if len(masks) == 0:
        return masks
    take = min(count, len(masks))
    indices = np.sort(rng.choice(len(masks), size=take, replace=False))
    return masks[indices]
