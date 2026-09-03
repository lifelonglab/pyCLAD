from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from pyclad.vision.strategies.replaycad.config import ReplayCADConfig

MECHANISM_VERSION = 2

_HASHED_FIELDS = (
    "model_id",
    "resolution",
    "condition_dim",
    "latent_scaling_factor",
    "semantic_tokens",
    "mask_group_width",
    "mask_projection_width",
    "use_spatial_conditioning",
    "compression_steps",
    "compression_batch_size",
    "base_learning_rate",
    "learning_rate_scale",
    "semantic_lr_multiplier",
    "weight_decay",
    "initializer_word",
    "prompt_templates",
    "timestep_sampling",
    "train_augmentation",
    "masks_per_concept",
    "mask_backend",
    "mask_modes",
    "sam_checkpoint",
    "sam_model_type",
    "precomputed_mask_root",
    "seed",
    "torch_dtype",
)


def concept_slug(concept_id: str) -> str:
    """Filesystem-safe directory name for a concept.

    Multidataset concepts arrive as ``<alias>__<category>`` (see
    ``pyclad/vision/data/multi_dataset.py``), and category names can contain spaces or slashes.
    """
    slug = re.sub(r"[\\/]+", "_", concept_id.strip())
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^A-Za-z0-9_.-]", "-", slug)
    if not slug:
        raise ValueError(f"Concept id {concept_id!r} does not produce a usable directory name")
    return slug


def compression_config_hash(config: ReplayCADConfig) -> str:
    payload = {"mechanism_version": MECHANISM_VERSION}
    for field in _HASHED_FIELDS:
        value = getattr(config, field)
        if isinstance(value, tuple):
            value = list(value)
        payload[field] = value
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class ConceptArtifact:
    """Everything ReplayCAD stores for one historical concept."""

    concept_id: str
    embedding: torch.Tensor
    projection_state: Optional[dict]
    masks: np.ndarray
    config_hash: str
    spatial_tokens: int


def save_artifact(artifact: ConceptArtifact, root: Path) -> Path:
    directory = Path(root) / concept_slug(artifact.concept_id)
    masks_dir = directory / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    (directory / "meta.json").unlink(missing_ok=True)

    for existing in masks_dir.glob("*.png"):
        existing.unlink()

    torch.save(artifact.embedding.detach().cpu(), directory / "embedding.pt")
    if artifact.projection_state is None:
        (directory / "projection.pt").unlink(missing_ok=True)
    else:
        torch.save(
            {key: value.detach().cpu() for key, value in artifact.projection_state.items()},
            directory / "projection.pt",
        )

    for index, mask in enumerate(artifact.masks):
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(masks_dir / f"{index:03d}.png")

    # Written to a temp file and renamed into place: meta.json is the crash-safety marker itself
    # (above), so a crash mid-write must not leave a truncated file that passes is_file() in
    # load_artifact() and then raises JSONDecodeError on every later run.
    meta_path = directory / "meta.json"
    meta_tmp_path = directory / "meta.json.tmp"
    meta_tmp_path.write_text(
        json.dumps(
            {
                "concept_id": artifact.concept_id,
                "config_hash": artifact.config_hash,
                "spatial_tokens": artifact.spatial_tokens,
                "mechanism_version": MECHANISM_VERSION,
                "semantic_tokens": int(artifact.embedding.shape[0]),
                "condition_dim": int(artifact.embedding.shape[1]),
                "mask_count": int(len(artifact.masks)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    os.replace(meta_tmp_path, meta_path)
    return directory


def load_artifact(concept_id: str, root: Path) -> Optional[ConceptArtifact]:
    directory = Path(root) / concept_slug(concept_id)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return None

    meta = json.loads(meta_path.read_text())
    embedding = torch.load(directory / "embedding.pt", map_location="cpu", weights_only=True)

    projection_path = directory / "projection.pt"
    projection_state = (
        torch.load(projection_path, map_location="cpu", weights_only=True) if projection_path.is_file() else None
    )

    mask_paths = sorted((directory / "masks").glob("*.png"))
    masks = (
        np.stack([np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in mask_paths])
        if mask_paths
        else np.zeros((0, 0, 0), dtype=np.uint8)
    )

    return ConceptArtifact(
        concept_id=meta["concept_id"],
        embedding=embedding,
        projection_state=projection_state,
        masks=masks,
        config_hash=meta["config_hash"],
        spatial_tokens=int(meta["spatial_tokens"]),
    )
