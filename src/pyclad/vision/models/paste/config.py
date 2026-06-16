from typing import Optional

from pydantic import Field, model_validator

from pyclad.vision.models.utilities.config import LightningVisionConfig


class PaSTeConfig(LightningVisionConfig):
    backbone_name: str = "mobilenet_v2"
    ad_layers: Optional[tuple[int, ...]] = None
    student_bootstrap_layer: Optional[int] = 0

    pretrained_teacher: bool = True
    pretrained_student: bool = False
    freeze_teacher: bool = True

    learning_rate: float = Field(default=0.4, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    @model_validator(mode="after")
    def _validate_bootstrap(self):
        if self.ad_layers is not None and len(self.ad_layers) == 0:
            raise ValueError("ad_layers must contain at least one layer index")
        if self.student_bootstrap_layer is not None and self.ad_layers is not None:
            if self.student_bootstrap_layer >= min(self.ad_layers):
                raise ValueError("student_bootstrap_layer must be smaller than the first anomaly-detection layer")
        return self
