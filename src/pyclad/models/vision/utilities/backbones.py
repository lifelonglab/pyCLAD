import inspect
from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn

_RESNET_RETURN_NODES = ["relu", "layer1", "layer2", "layer3", "layer4"]
_MOBILENET_V2_RETURN_NODES = ["features.1", "features.3", "features.6", "features.13", "features.18"]
_EFFICIENTNET_RETURN_NODES = ["features.1", "features.2", "features.3", "features.5", "features.7"]

_DEFAULT_RETURN_NODES: dict[str, list[str]] = {
    **{name: _RESNET_RETURN_NODES for name in ("resnet18", "resnet34", "resnet50", "wide_resnet50_2")},
    "mobilenet_v2": _MOBILENET_V2_RETURN_NODES,
    **{
        name: _EFFICIENTNET_RETURN_NODES
        for name in (
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_v2_s",
            "efficientnet_v2_m",
            "efficientnet_v2_l",
        )
    },
}


def default_backbone_return_nodes(backbone_name: str) -> list[str]:
    if backbone_name not in _DEFAULT_RETURN_NODES:
        raise ValueError(f"No default return nodes for '{backbone_name}'. Set backbone_return_nodes explicitly.")
    return list(_DEFAULT_RETURN_NODES[backbone_name])


def _resolve_torchvision_weights(tv_models, model_fn, backbone_name: str):
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
    import torchvision.models as tv_models

    model_fn = getattr(tv_models, backbone_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported backbone '{backbone_name}'")

    params = inspect.signature(model_fn).parameters
    if "weights" in params:
        weights = _resolve_torchvision_weights(tv_models, model_fn, backbone_name) if pretrained else None
        return model_fn(weights=weights)
    return model_fn(pretrained=pretrained)


class TorchvisionFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        return_nodes: Sequence[str],
        pretrained: bool,
        freeze: bool,
    ):
        super().__init__()

        if len(tuple(return_nodes)) == 0:
            raise ValueError("return_nodes must contain at least one feature node")

        from torchvision.models.feature_extraction import create_feature_extractor

        backbone = create_torchvision_model(backbone_name, pretrained)

        self._nodes = tuple(return_nodes)
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
        was_training = self._extractor.training
        self._extractor.eval()
        with torch.no_grad():
            try:
                device = next(self._extractor.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            x = torch.zeros((1, 3, input_size[0], input_size[1]), dtype=torch.float32, device=device)
            features = self._extractor(x)
        self._extractor.train(was_training)

        channels = tuple(int(features[name].shape[1]) for name in self._nodes)
        if len(channels) < 2:
            raise ValueError("Backbone encoder must expose at least 2 feature levels")
        return channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self._extractor(x)
        return [features[name] for name in self._nodes]
