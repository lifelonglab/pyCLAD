from typing import Optional

from pydantic import Field

from pyclad.vision.models.utilities.config import ImageSize, LightningVisionConfig


class FastFlowConfig(LightningVisionConfig):
    input_size: ImageSize = (256, 256)
    batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=200, ge=0)

    backbone_name: str = "wide_resnet50_2"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    # Torchvision weight variant loaded for the backbone, e.g. "IMAGENET1K_V2" or "DEFAULT".
    # Applies only where the backbone is actually pretrained -- it is ignored for randomly
    # initialised parts (any ``pretrained_*`` set to False). Spelled out rather than left implicit
    # so a serialised config records which weights produced a result. See ``utilities/backbones.py``
    # for why torchvision's own default is not followed.
    backbone_weights: str = "IMAGENET1K_V1"
    normalize_features: bool = True

    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)

    flow_steps: int = Field(default=8, gt=0)
    conv3x3_only: bool = False
    hidden_ratio: float = Field(default=1.0, gt=0.0)
    affine_clamping: float = Field(default=2.0, gt=0.0)
