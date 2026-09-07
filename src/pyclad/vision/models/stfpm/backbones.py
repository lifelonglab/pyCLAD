from __future__ import annotations

from pyclad.vision.models.utilities.backbones import (
    EFFICIENTNET_BACKBONES,
    MOBILENET_BACKBONES,
    RESNET_BACKBONES,
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
