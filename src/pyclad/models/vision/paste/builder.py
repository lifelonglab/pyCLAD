from pyclad.models.vision.paste.architecture import PaSTeArchitecture
from pyclad.models.vision.paste.config import PaSTeConfig


def build(config: PaSTeConfig) -> PaSTeArchitecture:
    return PaSTeArchitecture(
        backbone_name=config.backbone_name,
        ad_layers=config.ad_layers,
        student_bootstrap_layer=config.student_bootstrap_layer,
        pretrained_teacher=config.pretrained_teacher,
        pretrained_student=config.pretrained_student,
        freeze_teacher=config.freeze_teacher,
        input_size=config.input_size,
    )
