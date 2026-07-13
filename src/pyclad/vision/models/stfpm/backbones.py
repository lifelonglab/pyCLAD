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

_RESNET_RETURN_NODES = ["layer1", "layer2", "layer3"]
_MOBILENET_RETURN_NODES = ["features.3", "features.6", "features.13"]
_EFFICIENTNET_RETURN_NODES = ["features.2", "features.3", "features.5"]

_DEFAULT_RETURN_NODES: dict[str, list[str]] = {
    **{name: _RESNET_RETURN_NODES for name in RESNET_BACKBONES},
    **{name: _MOBILENET_RETURN_NODES for name in MOBILENET_BACKBONES},
    **{name: _EFFICIENTNET_RETURN_NODES for name in EFFICIENTNET_BACKBONES},
}


def supported_backbone_names() -> tuple[str, ...]:
    return tuple(_DEFAULT_RETURN_NODES.keys())


def default_stfpm_return_nodes(backbone_name: str) -> list[str]:
    if backbone_name not in _DEFAULT_RETURN_NODES:
        raise ValueError(
            f"No default return nodes for '{backbone_name}'. Supported presets: "
            f"{', '.join(supported_backbone_names())}. Set backbone_return_nodes explicitly to use "
            "a custom torchvision backbone."
        )
    return _DEFAULT_RETURN_NODES[backbone_name]


class TorchvisionFeatureExtractor(nn.Module):
    """
    Extracts a list of intermediate feature maps from a torchvision backbone by node name.
    """

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
