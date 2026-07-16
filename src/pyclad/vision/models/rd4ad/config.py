from typing import Literal

from pydantic import Field

from pyclad.vision.models.utilities.config import ImageSize, LightningVisionConfig


class RD4ADConfig(LightningVisionConfig):
    """Configuration for the RD4AD (Reverse Distillation from One-Class Embedding) detector."""

    input_size: ImageSize = (256, 256)
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=200, ge=0)
    learning_rate: float = Field(default=5e-3, gt=0.0)

    backbone_name: Literal["resnet18", "resnet34", "resnet50", "wide_resnet50_2"] = "wide_resnet50_2"
    pretrained_encoder: bool = True
    freeze_encoder: bool = True

    adam_beta1: float = Field(default=0.5, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)

    score_smoothing_sigma: float = Field(default=0.0, ge=0.0)
