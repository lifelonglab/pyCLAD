import inspect

import torchvision.models as tv_models
from torch import nn

RESNET_BACKBONES: tuple[str, ...] = ("resnet18", "resnet34", "resnet50", "wide_resnet50_2")
MOBILENET_BACKBONES: tuple[str, ...] = ("mobilenet_v2",)
EFFICIENTNET_BACKBONES: tuple[str, ...] = (
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
)
SUPPORTED_BACKBONES: tuple[str, ...] = RESNET_BACKBONES + MOBILENET_BACKBONES + EFFICIENTNET_BACKBONES


def _resolve_torchvision_weights(model_fn, backbone_name: str):
    get_model_weights = getattr(tv_models, "get_model_weights", None)
    if get_model_weights is not None:
        return get_model_weights(model_fn).DEFAULT

    attr_name = f"{backbone_name}_weights".lower()
    for attr in dir(tv_models):
        if attr.lower() == attr_name:
            enum_cls = getattr(tv_models, attr)
            if hasattr(enum_cls, "DEFAULT"):
                return enum_cls.DEFAULT
            if hasattr(enum_cls, "IMAGENET1K_V1"):
                return enum_cls.IMAGENET1K_V1
    return None


def create_torchvision_model(backbone_name: str, pretrained: bool) -> nn.Module:
    model_fn = getattr(tv_models, backbone_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported backbone '{backbone_name}'")

    params = inspect.signature(model_fn).parameters
    if "weights" in params:
        weights = _resolve_torchvision_weights(model_fn, backbone_name) if pretrained else None
        return model_fn(weights=weights)
    return model_fn(pretrained=pretrained)
