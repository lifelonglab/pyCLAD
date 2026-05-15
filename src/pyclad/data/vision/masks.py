from __future__ import annotations

import base64
import io
import json
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pyclad.data.vision.base import VisionSample


def load_ground_truth_mask(
    sample: VisionSample,
    resize_to: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    if sample.mask_path is None:
        height, width = resize_to if resize_to is not None else _image_size(sample.image_path)
        return np.zeros((height, width), dtype=np.uint8)

    mask_path = Path(sample.mask_path)
    if mask_path.suffix.lower() == ".json":
        mask = _load_annotation_mask(mask_path)
    else:
        mask = _load_bitmap_mask(mask_path)

    if resize_to is not None:
        mask = _resize_binary_mask(mask, resize_to)
    return mask.astype(np.uint8, copy=False)


def load_ground_truth_masks_for_samples(
    samples: Sequence[VisionSample],
    resize_to: Optional[tuple[int, int]] = None,
    skip_missing_anomaly_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    masks: list[np.ndarray] = []
    kept_indices: list[int] = []

    for index, sample in enumerate(samples):
        if sample.image_label == 1 and sample.mask_path is None:
            if skip_missing_anomaly_masks:
                continue
            raise FileNotFoundError(f"Missing anomaly mask for sample: {sample.image_path}")

        masks.append(load_ground_truth_mask(sample=sample, resize_to=resize_to))
        kept_indices.append(index)

    if not masks:
        empty_shape = (0, *(resize_to if resize_to is not None else (0, 0)))
        return np.zeros(empty_shape, dtype=np.uint8), np.asarray([], dtype=np.int64)

    return np.stack(masks, axis=0), np.asarray(kept_indices, dtype=np.int64)


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return height, width


def _load_bitmap_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"))
    return (mask > 0).astype(np.uint8, copy=False)


def _load_annotation_mask(annotation_path: Path) -> np.ndarray:
    annotation = json.loads(annotation_path.read_text())
    size = annotation.get("size", {})
    height = int(size.get("height", 0))
    width = int(size.get("width", 0))
    if height <= 0 or width <= 0:
        raise ValueError(f"Annotation does not define a valid size: {annotation_path}")

    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in annotation.get("objects", []):
        object_mask = _annotation_object_mask(obj=obj, canvas_shape=(height, width))
        mask = np.maximum(mask, object_mask)
    return mask


def _annotation_object_mask(obj: dict, canvas_shape: tuple[int, int]) -> np.ndarray:
    geometry_type = str(obj.get("geometryType", "")).lower()
    if geometry_type == "bitmap":
        return _supervisely_bitmap_mask(obj.get("bitmap"), canvas_shape)
    if geometry_type == "polygon":
        return _polygon_mask(obj.get("points"), canvas_shape)
    if geometry_type == "rectangle":
        return _rectangle_mask(obj.get("points"), canvas_shape)
    raise ValueError(f"Unsupported annotation geometry type: {obj.get('geometryType')!r}")


def _supervisely_bitmap_mask(bitmap: dict | None, canvas_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(bitmap, dict):
        raise ValueError("Bitmap annotation is missing the 'bitmap' payload")

    encoded = bitmap.get("data")
    origin = bitmap.get("origin")
    if not isinstance(encoded, str) or not isinstance(origin, list) or len(origin) != 2:
        raise ValueError("Bitmap annotation must define string 'data' and two-element 'origin'")

    decoded = zlib.decompress(base64.b64decode(encoded))
    with Image.open(io.BytesIO(decoded)) as bitmap_image:
        bitmap_array = np.asarray(bitmap_image)

    if bitmap_array.ndim == 3:
        if bitmap_array.shape[-1] >= 4:
            bitmap_array = bitmap_array[..., 3]
        else:
            bitmap_array = bitmap_array[..., 0]

    bitmap_mask = (bitmap_array > 0).astype(np.uint8, copy=False)

    origin_x, origin_y = int(origin[0]), int(origin[1])
    canvas_height, canvas_width = canvas_shape
    bitmap_height, bitmap_width = bitmap_mask.shape

    x0 = max(origin_x, 0)
    y0 = max(origin_y, 0)
    x1 = min(origin_x + bitmap_width, canvas_width)
    y1 = min(origin_y + bitmap_height, canvas_height)

    if x0 >= x1 or y0 >= y1:
        return np.zeros(canvas_shape, dtype=np.uint8)

    source_x0 = x0 - origin_x
    source_y0 = y0 - origin_y
    source_x1 = source_x0 + (x1 - x0)
    source_y1 = source_y0 + (y1 - y0)

    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas[y0:y1, x0:x1] = bitmap_mask[source_y0:source_y1, source_x0:source_x1]
    return canvas


def _polygon_mask(points: dict | None, canvas_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(points, dict):
        raise ValueError("Polygon annotation is missing the 'points' payload")

    width = int(canvas_shape[1])
    height = int(canvas_shape[0])
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)

    exterior = points.get("exterior", [])
    if exterior:
        draw.polygon([tuple(point) for point in exterior], fill=1)
    for interior in points.get("interior", []):
        draw.polygon([tuple(point) for point in interior], fill=0)

    return np.asarray(image, dtype=np.uint8)


def _rectangle_mask(points: dict | None, canvas_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(points, dict):
        raise ValueError("Rectangle annotation is missing the 'points' payload")

    exterior = points.get("exterior", [])
    if len(exterior) != 2:
        raise ValueError("Rectangle annotation must contain exactly two exterior points")

    width = int(canvas_shape[1])
    height = int(canvas_shape[0])
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([tuple(exterior[0]), tuple(exterior[1])], fill=1)
    return np.asarray(image, dtype=np.uint8)


def _resize_binary_mask(mask: np.ndarray, resize_to: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(resize_to):
        return mask.astype(np.uint8, copy=False)

    resampling_enum = getattr(Image, "Resampling", Image)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    resized = image.resize((resize_to[1], resize_to[0]), resampling_enum.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8, copy=False)
