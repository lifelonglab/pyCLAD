from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from pyclad.vision.models.utilities.backbones import (
    EFFICIENTNET_BACKBONES,
    MOBILENET_BACKBONES,
    RESNET_BACKBONES,
    create_torchvision_model,
)

_RESNET_RETURN_NODES = ["relu", "layer1", "layer2", "layer3", "layer4"]
_EFFICIENTNET_RETURN_NODES = ["features.1", "features.2", "features.3", "features.5", "features.7"]

_DEFAULT_RETURN_NODES: dict[str, list[str]] = {
    **{name: _RESNET_RETURN_NODES for name in RESNET_BACKBONES},
    **{name: ["features.1", "features.3", "features.6", "features.13", "features.18"] for name in MOBILENET_BACKBONES},
    **{name: _EFFICIENTNET_RETURN_NODES for name in EFFICIENTNET_BACKBONES},
}


def supported_backbone_names() -> tuple[str, ...]:
    return tuple(_DEFAULT_RETURN_NODES.keys())


def default_backbone_return_nodes(backbone_name: str) -> list[str]:
    if backbone_name not in _DEFAULT_RETURN_NODES:
        raise ValueError(f"No default return nodes for '{backbone_name}'. Set backbone_return_nodes explicitly.")
    return _DEFAULT_RETURN_NODES[backbone_name]


def default_fastflow_return_nodes(backbone_name: str) -> tuple[str, ...]:
    try:
        default_nodes = tuple(default_backbone_return_nodes(backbone_name))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported FastFlow backbone '{backbone_name}'. Supported default presets: "
            f"{', '.join(supported_backbone_names())}. You can still use a custom torchvision backbone by "
            "setting backbone_return_nodes explicitly."
        ) from exc

    # FastFlow is typically applied to a small set of intermediate 2D feature maps.
    if len(default_nodes) >= 4:
        return default_nodes[1:4]
    if len(default_nodes) >= 3:
        return default_nodes[:3]
    if len(default_nodes) >= 1:
        return default_nodes
    raise ValueError(f"Backbone '{backbone_name}' did not expose any default feature nodes")


class TorchvisionFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        return_nodes: Sequence[str],
        pretrained: bool,
        freeze: bool,
    ):
        super().__init__()

        self._nodes = tuple(return_nodes)
        if len(self._nodes) == 0:
            raise ValueError("return_nodes must contain at least one feature node")

        backbone = create_torchvision_model(backbone_name, pretrained=pretrained)
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
