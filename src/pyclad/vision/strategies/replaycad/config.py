import logging
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from pyclad.vision.strategies.replaycad.spatial import spatial_token_count

logger = logging.getLogger(__name__)

# Prompt templates from rinongal/textual_inversion (MIT, Rinon Gal et al.). ReplayCAD samples one
# uniformly per training step (personalized.py:218); generation instead uses a fixed prompt.
IMAGENET_TEMPLATES_SMALL: tuple[str, ...] = (
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    "an illustration of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "an illustration of a clean {}",
    "an illustration of a dirty {}",
    "a dark photo of the {}",
    "an illustration of my {}",
    "an illustration of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "an illustration of the {}",
    "a good photo of the {}",
    "an illustration of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "an illustration of the clean {}",
    "a rendition of a {}",
    "an illustration of a nice {}",
    "a good photo of a {}",
    "an illustration of the nice {}",
    "an illustration of the small {}",
    "an illustration of the weird {}",
    "an illustration of the large {}",
    "an illustration of a cool {}",
    "an illustration of a small {}",
    "a depiction of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a depiction of a clean {}",
    "a depiction of a dirty {}",
    "a dark photo of the {}",
    "a depiction of my {}",
    "a depiction of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a depiction of the {}",
    "a good photo of the {}",
    "a depiction of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a depiction of the clean {}",
    "a rendition of a {}",
    "a depiction of a nice {}",
    "a good photo of a {}",
    "a depiction of the nice {}",
    "a depiction of the small {}",
    "a depiction of the weird {}",
    "a depiction of the large {}",
    "a depiction of a cool {}",
    "a depiction of a small {}",
)

ReplayCADProfile = Literal["ldm-256", "sd-512"]
MaskBackendName = Literal["sam", "precomputed", "full-frame"]
MaskMode = Literal["object", "full-frame"]
MaskAugmentation = Literal[
    "none",
    "paper",
    "random_rotate",
    "random_3directions_rotate",
    "little_rotate_and_move",
    "visa_candle",
    "random_reset",
]
TrainAugmentation = Literal["none", "rotate_180", "rotate_3_directions", "rotate_all_directions"]


class ReplayCADConfig(BaseModel):
    """Faithful configuration of the ReplayCAD mechanism (Hu et al., IJCAI 2025).

    Defaults follow the paper's section 5.1 uniform profile. The authors' per-class tuning is
    reachable through ``pyclad.vision.strategies.replaycad.per_class``.
    """

    profile: ReplayCADProfile
    model_id: str
    resolution: int = Field(gt=0)
    condition_dim: int = Field(gt=0)
    latent_scaling_factor: float = Field(default=0.18215, gt=0.0)

    semantic_tokens: int = Field(default=20, gt=0)
    mask_group_width: int = Field(gt=0)
    mask_projection_width: int = Field(gt=0)
    use_spatial_conditioning: bool = True

    compression_steps: int = Field(gt=0)
    compression_batch_size: int = Field(gt=0)
    base_learning_rate: float = Field(default=5e-5, gt=0.0)
    learning_rate_scale: int = Field(gt=0)
    semantic_lr_multiplier: float = Field(default=100.0, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    initializer_word: Optional[str] = None
    train_augmentation: TrainAugmentation = "none"
    prompt_templates: tuple[str, ...] = IMAGENET_TEMPLATES_SMALL
    generation_prompt: str = "a photo of *"
    timestep_sampling: Literal["uniform", "cubic"] = "uniform"

    inference_steps: int = Field(default=50, gt=0)
    ddim_eta: float = Field(default=0.0, ge=0.0)
    guidance_scale: float = Field(default=10.0, ge=1.0)
    replay_samples_per_concept: int = Field(default=800, gt=0)
    generation_batch_size: int = Field(default=8, gt=0)

    masks_per_concept: int = Field(default=10, ge=1, le=10)
    mask_backend: MaskBackendName = "sam"
    sam_checkpoint: Optional[Path] = None
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    precomputed_mask_root: Optional[Path] = None
    mask_modes: Dict[str, MaskMode] = Field(default_factory=dict)
    mask_augmentation: MaskAugmentation = "none"
    max_mask_rotation_degrees: float = Field(default=360.0, ge=0.0)
    max_mask_shift_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    mask_transform_angles: int = Field(default=2, ge=0)
    mask_transform_distance: float = Field(default=0.05, ge=0.0)
    mask_transform_transpose: bool = False
    visa_candle_shift_pixels: int = Field(default=20, ge=0)
    visa_candle_rotate: bool = False

    artifact_dir: Path
    strict_cache: bool = False

    device: str = "cuda"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    local_files_only: bool = False
    seed: int = 42

    @property
    def latent_size(self) -> int:
        """Number of values in the four-channel VAE latent of one mask."""
        return (self.resolution // 8) * (self.resolution // 8) * 4

    @property
    def spatial_tokens(self) -> int:
        return spatial_token_count(
            self.latent_size, self.mask_group_width, self.mask_projection_width, self.condition_dim
        )

    @property
    def spatial_learning_rate(self) -> float:
        return self.base_learning_rate * self.learning_rate_scale

    @property
    def semantic_learning_rate(self) -> float:
        return self.spatial_learning_rate * self.semantic_lr_multiplier

    @model_validator(mode="after")
    def _validate_shapes(self):
        if self.latent_size % self.mask_group_width:
            raise ValueError(
                "The flattened four-channel mask latent must be divisible by mask_group_width: "
                f"{self.latent_size} % {self.mask_group_width} != 0"
            )
        projected = (self.latent_size // self.mask_group_width) * self.mask_projection_width
        if projected % self.condition_dim:
            raise ValueError(
                "Projected mask features cannot be reshaped into conditioning tokens: "
                f"{projected} % {self.condition_dim} != 0. Adjust mask_group_width "
                f"({self.mask_group_width}) or mask_projection_width ({self.mask_projection_width})."
            )
        if self.mask_backend == "sam" and self.sam_checkpoint is None:
            raise ValueError("sam_checkpoint is required when mask_backend='sam'")
        if self.mask_backend == "precomputed" and self.precomputed_mask_root is None:
            raise ValueError("precomputed_mask_root is required when mask_backend='precomputed'")
        return self

    @classmethod
    def for_benchmark(cls, benchmark: str, artifact_dir: Path, **overrides) -> "ReplayCADConfig":
        """Build the paper's uniform profile for a benchmark.

        MVTec AD uses the LDM-256 profile and VisA the SD-512 profile, as in section 5.1. Every
        other benchmark (BTech, DAGM, MPDD, multidataset streams) uses the LDM-256 profile with a
        neutral initializer, since the authors publish no configuration for them.
        """
        presets = {
            "visa": dict(
                profile="sd-512",
                model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                resolution=512,
                condition_dim=768,
                mask_group_width=128,
                mask_projection_width=192,
                compression_steps=30_000,
                compression_batch_size=2,
                learning_rate_scale=4,
                generation_batch_size=2,
            ),
        }
        ldm_256 = dict(
            profile="ldm-256",
            model_id="CompVis/ldm-text2im-large-256",
            resolution=256,
            condition_dim=1280,
            mask_group_width=128,
            mask_projection_width=200,
            compression_steps=20_000,
            compression_batch_size=16,
            learning_rate_scale=32,
            generation_batch_size=8,
        )
        preset = presets.get(benchmark.lower(), ldm_256)
        if benchmark.lower() not in presets and benchmark.lower() != "mvtec":
            logger.info(
                "ReplayCAD has no published profile for benchmark '%s'; using the MVTec LDM-256 "
                "profile with a neutral initializer",
                benchmark,
            )
        return cls(artifact_dir=artifact_dir, **{**preset, **overrides})
