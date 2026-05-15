from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from pyclad.models.vision.utilities.backbones import create_torchvision_model


@dataclass(frozen=True)
class PaSTeBackboneSpec:
    default_ad_layers: tuple[int, ...]
    stage_builder: Callable[[nn.Module], list[nn.Module]]


def _resnet_stages(model: nn.Module) -> list[nn.Module]:
    return [
        nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]


def _features_stages(model: nn.Module) -> list[nn.Module]:
    return list(model.features.children())


_RESNET_SPEC = PaSTeBackboneSpec(default_ad_layers=(1, 2, 3), stage_builder=_resnet_stages)
_MOBILENET_V2_SPEC = PaSTeBackboneSpec(default_ad_layers=(3, 6, 13), stage_builder=_features_stages)
_EFFICIENTNET_SPEC = PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages)

_BACKBONE_SPECS: dict[str, PaSTeBackboneSpec] = {
    **{name: _RESNET_SPEC for name in ("resnet18", "resnet34", "resnet50", "wide_resnet50_2")},
    "mobilenet_v2": _MOBILENET_V2_SPEC,
    **{
        name: _EFFICIENTNET_SPEC
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


def supported_backbone_names() -> tuple[str, ...]:
    return tuple(_BACKBONE_SPECS.keys())


def default_ad_layers(backbone_name: str) -> tuple[int, ...]:
    return resolve_backbone_spec(backbone_name).default_ad_layers


def resolve_backbone_spec(backbone_name: str) -> PaSTeBackboneSpec:
    try:
        return _BACKBONE_SPECS[backbone_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Supported backbones: {', '.join(supported_backbone_names())}"
        ) from exc


def supported_stage_indices(backbone_name: str, pretrained: bool = False) -> tuple[int, ...]:
    spec = resolve_backbone_spec(backbone_name)
    model = create_torchvision_model(backbone_name, pretrained=pretrained)
    stages = spec.stage_builder(model)
    return tuple(range(len(stages)))


class PaSTeBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        ad_layers: tuple[int, ...],
        pretrained: bool,
        freeze: bool,
        bootstrap_layer: int | None,
        is_teacher: bool,
    ):
        super().__init__()

        spec = resolve_backbone_spec(backbone_name)
        stages = spec.stage_builder(create_torchvision_model(backbone_name, pretrained=pretrained))
        max_layer = max(ad_layers)

        if max_layer >= len(stages):
            raise ValueError(
                f"Requested ad layer {max_layer} for backbone '{backbone_name}', but it has only "
                f"{len(stages)} stages indexed from 0 to {len(stages) - 1}"
            )

        self.backbone_name = backbone_name
        self.ad_layers = tuple(sorted(ad_layers))
        self.bootstrap_layer = bootstrap_layer
        self.is_teacher = is_teacher

        if is_teacher:
            stage_slice = slice(0, max_layer + 1)
            self.layer_offset = 0
        else:
            start_layer = 0 if bootstrap_layer is None else bootstrap_layer + 1
            stage_slice = slice(start_layer, max_layer + 1)
            self.layer_offset = start_layer

        self.stages = nn.ModuleList(stages[stage_slice])

        if freeze:
            for parameter in self.stages.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        features: list[torch.Tensor] = []
        bootstrap_feature: torch.Tensor | None = None

        for layer_index, stage in enumerate(self.stages, start=self.layer_offset):
            x = stage(x)
            if layer_index in self.ad_layers:
                features.append(x)
            if self.bootstrap_layer is not None and layer_index == self.bootstrap_layer:
                bootstrap_feature = x.clone()

        return features, bootstrap_feature
