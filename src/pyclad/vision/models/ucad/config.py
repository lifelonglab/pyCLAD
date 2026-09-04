from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator

from pyclad.vision.models.utilities.config import ImageSize, VisionConfig


class UCADConfig(VisionConfig):
    """Configuration for UCAD.

    Defaults follow the reference implementation (https://github.com/shirowalker/UCAD),
    not the paper prose, wherever the two disagree. Each divergence is recorded in
    ``docs/vision.md``. Training fields live here rather than in ``LightningVisionConfig``
    because UCAD optimises a single prompt tensor for a fixed number of epochs with no
    validation split and no early stopping, so that base class's fields would be inert.
    """

    input_size: ImageSize = (224, 224)
    batch_size: int = Field(default=8, gt=0)

    # timm 0.6.7 resolved `vit_base_patch16_224` + `pretrained=True` to these weights
    # (ImageNet-21k pretrained, ImageNet-1k finetuned). The paper's text claims plain
    # ImageNet-21k; `vit_base_patch16_224.augreg_in21k` selects that as an ablation.
    backbone_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k"
    pretrained_backbone: bool = True

    feature_layer: int = Field(default=6, gt=0)
    prompt_depth: int = Field(default=6, gt=0)
    prompt_length: int = Field(default=1, gt=0)
    prompt_init: Literal["zero", "uniform"] = "uniform"

    target_embed_dimension: int = Field(default=1024, gt=0)
    key_size: int = Field(default=196, gt=0)
    knowledge_size: int = Field(default=196, gt=0)
    coreset_projection_dimension: int = Field(default=128, gt=0)
    coreset_starting_points: int = Field(default=10, gt=0)
    n_neighbors: int = Field(default=1, gt=0)

    epochs: int = Field(default=25, ge=0)
    learning_rate: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    contrastive_temperature: float = Field(default=0.5, gt=0.0)
    show_training_progress: bool = False

    structure_mode: Literal["none", "precomputed", "sam"] = "none"
    structure_mask_root: Optional[str] = None
    mask_interpolation: Literal["bilinear", "nearest"] = "bilinear"
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_b"

    routing_statistic: Literal["mean", "max"] = "mean"
    smoothing_sigma: float = Field(default=4.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_ucad_configuration(self):
        if self.structure_mode == "precomputed":
            if not self.structure_mask_root:
                raise ValueError("structure_mask_root is required when structure_mode='precomputed'")
            if not Path(self.structure_mask_root).expanduser().exists():
                raise ValueError(f"structure_mask_root does not exist: {self.structure_mask_root}")
        if self.structure_mode == "sam":
            if not self.sam_checkpoint:
                raise ValueError("sam_checkpoint is required when structure_mode='sam'")
            if not Path(self.sam_checkpoint).expanduser().is_file():
                raise ValueError(f"sam_checkpoint does not exist: {self.sam_checkpoint}")
        return self
