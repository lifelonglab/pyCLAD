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
    # Torchvision weight variant loaded for the backbone, e.g. "IMAGENET1K_V2" or "DEFAULT".
    # Applies only where the backbone is actually pretrained -- it is ignored for randomly
    # initialised parts (any ``pretrained_*`` set to False). Spelled out rather than left implicit
    # so a serialised config records which weights produced a result. See ``utilities/backbones.py``
    # for why torchvision's own default is not followed.
    backbone_weights: str = "IMAGENET1K_V1"

    learning_rate: float = Field(default=0.4, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
