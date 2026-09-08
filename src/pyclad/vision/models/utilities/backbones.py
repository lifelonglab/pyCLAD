import inspect
from collections import OrderedDict
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


def _weights_enum(model_fn, backbone_name: str):
    get_model_weights = getattr(tv_models, "get_model_weights", None)
    if get_model_weights is not None:
        # By registered name, not model_fn: torchvision resolves via the `weights` parameter's
        # type annotation, which breaks for callers (e.g. tests) using an unannotated stand-in.
        return get_model_weights(backbone_name)

    attr_name = f"{backbone_name}_weights".lower()
    for attr in dir(tv_models):
        if attr.lower() == attr_name:
            return getattr(tv_models, attr)
    return None


def _resolve_torchvision_weights(model_fn, backbone_name: str, weights_name: Optional[str]):
    enum_cls = _weights_enum(model_fn, backbone_name)
    if enum_cls is None:
        return None

    if weights_name is None:
        if hasattr(enum_cls, "DEFAULT"):
            return enum_cls.DEFAULT
        return getattr(enum_cls, "IMAGENET1K_V1", None)

    if not hasattr(enum_cls, weights_name):
        available = ", ".join(member.name for member in enum_cls)
        raise ValueError(
            f"Unknown pretrained weights '{weights_name}' for backbone '{backbone_name}'. Available: {available}"
        )
    return getattr(enum_cls, weights_name)


def create_torchvision_model(
    backbone_name: str,
    pretrained: bool,
    weights_name: Optional[str] = None,
) -> nn.Module:
    """Build a torchvision backbone.

    ``weights_name`` selects an explicit member of the backbone's weight enum, e.g.
    ``"IMAGENET1K_V1"``. Leave it ``None`` to keep torchvision's ``DEFAULT``, which is a moving
    pointer: for ``wide_resnet50_2`` it currently resolves to ``IMAGENET1K_V2``, while reference
    anomaly-detection implementations load the ``IMAGENET1K_V1`` checkpoint
    (``wide_resnet50_2-95faca4d.pth``).
    """
    model_fn = getattr(tv_models, backbone_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported backbone '{backbone_name}'")

    params = inspect.signature(model_fn).parameters
    if "weights" in params:
        weights = _resolve_torchvision_weights(model_fn, backbone_name, weights_name) if pretrained else None
        return model_fn(weights=weights)
    return model_fn(pretrained=pretrained)


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


class TorchvisionFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        return_nodes: Sequence[str],
        pretrained: bool,
        freeze: bool,
        weights_name: Optional[str] = None,
    ):
        super().__init__()

        self._nodes = tuple(return_nodes)
        if len(self._nodes) == 0:
            raise ValueError("return_nodes must contain at least one feature node")

        backbone = create_torchvision_model(backbone_name, pretrained=pretrained, weights_name=weights_name)
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

    def infer_out_channels(self, input_size: tuple[int, int]) -> tuple[int, ...]:
        """Channel count of each returned feature map, probed with a single dummy forward."""
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            with torch.no_grad():
                features = self.forward(torch.zeros(1, 3, input_size[0], input_size[1], device=device))
        finally:
            self.train(was_training)
        return tuple(feature.shape[1] for feature in features)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self._extractor(x)
        return [features[name] for name in self._nodes]
