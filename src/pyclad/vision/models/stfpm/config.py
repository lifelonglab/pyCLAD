from typing import Optional

from pydantic import Field

from pyclad.vision.models.utilities.config import ImageSize, LightningVisionConfig


class STFPMConfig(LightningVisionConfig):

    input_size: ImageSize = (256, 256)
    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=100, ge=0)

    backbone_name: str = "resnet18"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_teacher: bool = True
    pretrained_student: bool = False
    freeze_teacher: bool = True

    learning_rate: float = Field(default=0.4, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
