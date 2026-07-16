"""ResNet teacher encoder + OCBE bottleneck for RD4AD."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class AttnBasicBlock(BasicBlock):
    pass


class AttnBottleneck(Bottleneck):
    pass


@dataclass(frozen=True)
class RD4ADBackboneSpec:
    name: str
    constructor_name: str
    bn_block_cls: type[nn.Module]
    bn_blocks: int
    decoder_layers: tuple[int, int, int]
    expansion: int
    groups: int = 1
    width_per_group: int = 64

    @property
    def feature_channels(self) -> tuple[int, int, int]:
        return (64 * self.expansion, 128 * self.expansion, 256 * self.expansion)

    @property
    def bottleneck_channels(self) -> int:
        return 512 * self.expansion


_BACKBONE_SPECS = {
    "resnet18": RD4ADBackboneSpec(
        name="resnet18",
        constructor_name="resnet18",
        bn_block_cls=AttnBasicBlock,
        bn_blocks=2,
        decoder_layers=(2, 2, 2),
        expansion=1,
    ),
    "resnet34": RD4ADBackboneSpec(
        name="resnet34",
        constructor_name="resnet34",
        bn_block_cls=AttnBasicBlock,
        bn_blocks=3,
        decoder_layers=(3, 4, 6),
        expansion=1,
    ),
    "resnet50": RD4ADBackboneSpec(
        name="resnet50",
        constructor_name="resnet50",
        bn_block_cls=AttnBottleneck,
        bn_blocks=3,
        decoder_layers=(3, 4, 6),
        expansion=4,
    ),
    "wide_resnet50_2": RD4ADBackboneSpec(
        name="wide_resnet50_2",
        constructor_name="wide_resnet50_2",
        bn_block_cls=AttnBottleneck,
        bn_blocks=3,
        decoder_layers=(3, 4, 6),
        expansion=4,
        width_per_group=128,
    ),
}


def supported_backbone_names() -> tuple[str, ...]:
    return tuple(_BACKBONE_SPECS.keys())


def resolve_backbone_spec(backbone_name: str) -> RD4ADBackboneSpec:
    if backbone_name not in _BACKBONE_SPECS:
        raise ValueError(
            f"Unsupported RD4AD backbone '{backbone_name}'. Supported backbones: {', '.join(supported_backbone_names())}"
        )
    return _BACKBONE_SPECS[backbone_name]


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)
        return [feature_a, feature_b, feature_c]


class BNLayer(nn.Module):
    def __init__(
        self,
        backbone_spec: RD4ADBackboneSpec,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self._block_cls = backbone_spec.bn_block_cls
        self.groups = backbone_spec.groups
        self.base_width = backbone_spec.width_per_group
        self.inplanes = 256 * self._block_cls.expansion
        self.dilation = 1

        channel_a, channel_b, _ = backbone_spec.feature_channels
        reduced_channel_b = 128 * self._block_cls.expansion
        reduced_channel_c = 256 * self._block_cls.expansion

        self.conv1 = conv3x3(channel_a, reduced_channel_b, stride=2)
        self.bn1 = norm_layer(reduced_channel_b)
        self.conv2 = conv3x3(reduced_channel_b, reduced_channel_c, stride=2)
        self.bn2 = norm_layer(reduced_channel_c)
        self.conv3 = conv3x3(channel_b, reduced_channel_c, stride=2)
        self.bn3 = norm_layer(reduced_channel_c)
        self.relu = nn.ReLU(inplace=True)
        self.bn_layer = self._make_layer(self._block_cls, 512, backbone_spec.bn_blocks, stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes * 3,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                norm_layer=norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: list[Tensor]) -> Tensor:
        if len(x) != 3:
            raise ValueError(f"BNLayer expects 3 encoder features, got {len(x)}")

        l1 = self.relu(self.bn1(self.conv1(x[0])))
        l1 = self.relu(self.bn2(self.conv2(l1)))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], dim=1)
        return self.bn_layer(feature).contiguous()


def _resolve_torchvision_weights(tv_models, model_fn):
    get_model_weights = getattr(tv_models, "get_model_weights", None)
    if get_model_weights is not None:
        enum_cls = get_model_weights(model_fn)
        return getattr(enum_cls, "IMAGENET1K_V1", enum_cls.DEFAULT)

    for attr in dir(tv_models):
        if attr.lower() == f"{model_fn.__name__}_weights".lower():
            enum_cls = getattr(tv_models, attr)
            if hasattr(enum_cls, "IMAGENET1K_V1"):
                return enum_cls.IMAGENET1K_V1
            if hasattr(enum_cls, "DEFAULT"):
                return enum_cls.DEFAULT
    return None


def build_encoder_and_bn(backbone_name: str, pretrained: bool) -> tuple[ResNetEncoder, BNLayer]:
    import torchvision.models as tv_models

    backbone_spec = resolve_backbone_spec(backbone_name)
    model_fn = getattr(tv_models, backbone_spec.constructor_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported torchvision backbone '{backbone_spec.constructor_name}'")

    parameters = inspect.signature(model_fn).parameters
    if "weights" in parameters:
        backbone = model_fn(weights=_resolve_torchvision_weights(tv_models, model_fn) if pretrained else None)
    else:
        backbone = model_fn(pretrained=pretrained)

    return ResNetEncoder(backbone=backbone), BNLayer(backbone_spec=backbone_spec)
