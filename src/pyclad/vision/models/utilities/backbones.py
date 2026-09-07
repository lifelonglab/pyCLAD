from __future__ import annotations

import inspect
from collections import OrderedDict
from enum import Enum
from typing import Optional, Sequence

import torch
import torchvision.models as tv_models
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

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

_PINNED_WEIGHTS = "IMAGENET1K_V1"


def _torchvision_model_fn(backbone_name: str):
    model_fn = getattr(tv_models, backbone_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported backbone '{backbone_name}'")
    return model_fn


def _weights_enum(model_fn, backbone_name: str):
    get_model_weights = getattr(tv_models, "get_model_weights", None)
    if get_model_weights is not None:
        return get_model_weights(model_fn)

    attr_name = f"{backbone_name}_weights".lower()
    for attr in dir(tv_models):
        if attr.lower() == attr_name:
            return getattr(tv_models, attr)
    return None


def resolve_pretrained_weights(backbone_name: str, weights: Optional[str] = None) -> Optional[Enum]:
    """Resolve a torchvision weights enum member by name.

    ``weights=None`` selects the pinned ImageNet V1 checkpoint (see ``_PINNED_WEIGHTS``); any enum
    member name, or ``"DEFAULT"``, may be requested instead. Returns ``None`` only for torchvision
    versions old enough to have no weight enums at all.
    """
    enum_cls = _weights_enum(_torchvision_model_fn(backbone_name), backbone_name)
    if enum_cls is None:
        return None

    resolved = getattr(enum_cls, weights or _PINNED_WEIGHTS, None)
    if isinstance(resolved, enum_cls):
        return resolved

    if weights is None:
        return getattr(enum_cls, "DEFAULT", None)

    available = ", ".join(member.name for member in enum_cls)
    raise ValueError(
        f"Unknown pretrained weights '{weights}' for backbone '{backbone_name}'. "
        f"Available variants: {available}, or 'DEFAULT' for torchvision's own choice."
    )


def create_torchvision_model(backbone_name: str, pretrained: bool, weights: Optional[str] = None) -> nn.Module:
    model_fn = _torchvision_model_fn(backbone_name)

    if "weights" not in inspect.signature(model_fn).parameters:
        if weights is not None:
            raise ValueError(
                f"Selecting pretrained weights '{weights}' requires torchvision>=0.13; "
                "the installed version only supports the legacy pretrained=True flag."
            )
        return model_fn(pretrained=pretrained)

    if not pretrained:
        return model_fn(weights=None)

    return model_fn(weights=resolve_pretrained_weights(backbone_name, weights))


class TorchvisionFeatureExtractor(nn.Module):
    """Extracts a list of intermediate feature maps from a torchvision backbone by node name."""

    def __init__(
        self,
        backbone_name: str,
        return_nodes: Sequence[str],
        pretrained: bool,
        freeze: bool,
        weights: Optional[str] = None,
    ):
        super().__init__()

        self._nodes = tuple(return_nodes)
        if len(self._nodes) == 0:
            raise ValueError("return_nodes must contain at least one feature node")

        backbone = create_torchvision_model(backbone_name, pretrained=pretrained, weights=weights)
        self._extractor = create_feature_extractor(
            model=backbone,
            return_nodes=OrderedDict((node, node) for node in self._nodes),
        )

        if freeze:
            for parameter in self._extractor.parameters():
                parameter.requires_grad = False

    @property
    def return_nodes(self) -> tuple[str, ...]:
        return self._nodes

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self._extractor(x)
        return [features[name] for name in self._nodes]
