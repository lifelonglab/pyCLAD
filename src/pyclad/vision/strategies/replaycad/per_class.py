"""The ReplayCAD authors' hand-tuned per-class settings, as an opt-in override table.

The paper (section 5.1) describes one uniform configuration per dataset; ``ReplayCADConfig.for_benchmark``
follows that. The scripts the authors released instead hand-tune each class (backbone, projection
width, steps, guidance, sample count) and disable spatial conditioning entirely for two of them.
``apply_per_class`` returns that alternative, re-validated profile for one concept -- a reference
table, not a runnable mode: ``ReplayCADMemory``/``DiffusersBackend`` each hold a single config for
the whole stream, while this table varies ``model_id``, ``condition_dim`` and ``resolution`` per
class, so driving a live run through it would fail on a shape mismatch (VisA alone mixes the
LDM-256 and SD1.5-512 families).

Values were extracted mechanically from ``tran_mvtec.sh``, ``train_visa.sh``, ``generate_mvtec.sh``,
``generate_visa.sh`` and the YAML configs they reference. Three source inconsistencies, recorded
rather than silently resolved:

1. Several generation checkpoints exceed their training config's ``max_steps`` (e.g. cable
   ``gs-5999`` vs. ``max_steps: 5000``). ``compression_steps`` below uses the checkpoint step, since
   that is what produced the published samples.
2. ``train_visa.sh`` passes ``--init_word Process Control Block`` to a single-value argparse option;
   only the first word is kept, so the effective initializer is ``"Process"``, not the full phrase.
3. ``capsules`` and ``macaroni2`` are generated through maskless scripts, so ``mask_augmentation``
   stays ``"none"`` for both -- not either of the two release dispatchers' dead,
   mutually-contradictory ``macaroni2`` branches (``random_reset`` vs. ``random_rotate``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal

from pyclad.vision.strategies.replaycad.config import (
    MaskAugmentation,
    ReplayCADConfig,
    TrainAugmentation,
)
from pyclad.vision.strategies.replaycad.masks import category_of

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerClassOverride:
    """One class's settings as published in the ReplayCAD release scripts."""

    profile: Literal["ldm-256", "sd-512"]
    compression_steps: int
    guidance_scale: float
    replay_samples: int
    initializer_word: str
    train_augmentation: TrainAugmentation
    timestep_sampling: Literal["uniform", "cubic"] = "uniform"
    use_spatial_conditioning: bool = True
    wide_projection: bool = False
    semantic_tokens: int = 20
    base_learning_rate: float = 5e-5
    semantic_lr_multiplier: float = 100.0
    mask_augmentation: MaskAugmentation = "none"
    mask_transform_angles: int = 2
    mask_transform_distance: float = 0.05
    mask_transform_transpose: bool = False
    visa_candle_shift_pixels: int = 20
    visa_candle_rotate: bool = False


_M = PerClassOverride

MVTEC_OVERRIDES: Dict[str, PerClassOverride] = {
    "bottle": _M("ldm-256", 2499, 10.0, 200, "screw", "none"),
    "cable": _M("ldm-256", 5999, 10.0, 400, "screw", "none"),
    "capsule": _M("ldm-256", 14999, 10.0, 200, "screw", "none"),
    "carpet": _M("ldm-256", 9999, 1.0, 200, "screw", "none"),
    "grid": _M(
        "ldm-256", 24999, 5.0, 800, "screw", "rotate_3_directions", mask_augmentation="random_3directions_rotate"
    ),
    "hazelnut": _M("ldm-256", 19999, 10.0, 200, "screw", "rotate_all_directions", mask_augmentation="random_rotate"),
    "leather": _M("ldm-256", 3499, 1.0, 200, "screw", "none"),
    "metal_nut": _M("ldm-256", 19999, 10.0, 800, "screw", "rotate_all_directions", mask_augmentation="random_rotate"),
    "pill": _M("ldm-256", 1499, 10.0, 200, "screw", "none"),
    "screw": _M("ldm-256", 19999, 10.0, 800, "screw", "rotate_all_directions", mask_augmentation="random_rotate"),
    "tile": _M("ldm-256", 4999, 10.0, 200, "screw", "none"),
    "toothbrush": _M("ldm-256", 1499, 10.0, 200, "screw", "none"),
    "transistor": _M("ldm-256", 4999, 10.0, 800, "screw", "none", mask_augmentation="little_rotate_and_move"),
    "wood": _M("ldm-256", 5999, 1.0, 200, "screw", "none"),
}

VISA_OVERRIDES: Dict[str, PerClassOverride] = {
    "candle": _M("ldm-256", 9999, 10.0, 800, "screw", "none", mask_augmentation="visa_candle"),
    "capsules": _M(
        "sd-512",
        52499,
        10.0,
        400,
        "Process",
        "rotate_180",
        use_spatial_conditioning=False,
        semantic_tokens=40,
        base_learning_rate=5e-3,
        semantic_lr_multiplier=1.0,
    ),
    "cashew": _M(
        "sd-512",
        29999,
        10.0,
        400,
        "Process",
        "none",
        mask_augmentation="little_rotate_and_move",
        mask_transform_angles=10,
    ),
    "chewinggum": _M(
        "sd-512",
        29999,
        10.0,
        400,
        "Process",
        "rotate_180",
        mask_augmentation="little_rotate_and_move",
        mask_transform_distance=0.1,
        mask_transform_transpose=True,
    ),
    "fryum": _M(
        "sd-512",
        31656,
        10.0,
        400,
        "Process",
        "rotate_180",
        "cubic",
        wide_projection=True,
        mask_augmentation="random_rotate",
    ),
    "macaroni1": _M(
        "ldm-256", 9999, 10.0, 800, "screw", "none", mask_augmentation="visa_candle", visa_candle_rotate=True
    ),
    "macaroni2": _M(
        "ldm-256",
        14535,
        10.0,
        400,
        "candle",
        "none",
        use_spatial_conditioning=False,
        semantic_tokens=40,
        base_learning_rate=5e-3,
        semantic_lr_multiplier=1.0,
    ),
    "pcb1": _M(
        "sd-512",
        26499,
        10.0,
        400,
        "Process",
        "rotate_180",
        "cubic",
        mask_augmentation="little_rotate_and_move",
        mask_transform_distance=0.02,
        mask_transform_transpose=True,
    ),
    "pcb2": _M(
        "sd-512",
        29999,
        10.0,
        400,
        "Process",
        "rotate_180",
        "cubic",
        mask_augmentation="little_rotate_and_move",
        mask_transform_distance=0.02,
        mask_transform_transpose=True,
    ),
    "pcb3": _M(
        "sd-512",
        29499,
        10.0,
        400,
        "Process",
        "rotate_180",
        "cubic",
        mask_augmentation="little_rotate_and_move",
        mask_transform_distance=0.02,
        mask_transform_transpose=True,
    ),
    "pcb4": _M(
        "sd-512",
        29999,
        10.0,
        400,
        "Process",
        "none",
        "cubic",
        wide_projection=True,
        mask_augmentation="little_rotate_and_move",
        mask_transform_distance=0.02,
    ),
    # pipe_fryum is absent from both release scripts.
}

AUTHORS_OVERRIDES: Dict[str, Dict[str, PerClassOverride]] = {
    "mvtec": MVTEC_OVERRIDES,
    "visa": VISA_OVERRIDES,
}


def _profile_fields(profile: str, wide_projection: bool) -> dict:
    if profile == "sd-512":
        return {
            "profile": "sd-512",
            "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "resolution": 512,
            "condition_dim": 768,
            "mask_group_width": 512 if wide_projection else 128,
            "mask_projection_width": 192,
            "compression_batch_size": 2,
            "learning_rate_scale": 4,
            "generation_batch_size": 2,
        }
    return {
        "profile": "ldm-256",
        "model_id": "CompVis/ldm-text2im-large-256",
        "resolution": 256,
        "condition_dim": 1280,
        "mask_group_width": 256 if wide_projection else 128,
        "mask_projection_width": 400 if wide_projection else 200,
        "compression_batch_size": 16,
        "learning_rate_scale": 32,
        "generation_batch_size": 8,
    }


def apply_per_class(config: ReplayCADConfig, concept_id: str, benchmark: str) -> ReplayCADConfig:
    """Return ``config`` with this concept's published settings applied.

    Benchmarks the authors did not publish, and classes missing from their scripts, keep the
    dataset default and are logged.
    """
    table = AUTHORS_OVERRIDES.get(benchmark.lower())
    if table is None:
        return config

    override = table.get(category_of(concept_id))
    if override is None:
        logger.warning(
            "ReplayCAD has no published per-class settings for '%s' in %s; using the dataset default profile",
            concept_id,
            benchmark,
        )
        return config

    updates = _profile_fields(override.profile, override.wide_projection)
    updates.update(
        compression_steps=override.compression_steps,
        guidance_scale=override.guidance_scale,
        replay_samples_per_concept=override.replay_samples,
        initializer_word=override.initializer_word,
        timestep_sampling=override.timestep_sampling,
        train_augmentation=override.train_augmentation,
        use_spatial_conditioning=override.use_spatial_conditioning,
        semantic_tokens=override.semantic_tokens,
        base_learning_rate=override.base_learning_rate,
        semantic_lr_multiplier=override.semantic_lr_multiplier,
        mask_augmentation=override.mask_augmentation,
        mask_transform_angles=override.mask_transform_angles,
        mask_transform_distance=override.mask_transform_distance,
        mask_transform_transpose=override.mask_transform_transpose,
        visa_candle_shift_pixels=override.visa_candle_shift_pixels,
        visa_candle_rotate=override.visa_candle_rotate,
    )

    return type(config).model_validate({**config.model_dump(), **updates})
