"""Reverse-ResNet student decoder (de_resnet) for RD4AD."""

from __future__ import annotations

from typing import Callable, Optional

from torch import Tensor, nn

from pyclad.vision.models.rd4ad.standard.resnet import resolve_backbone_spec


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


def deconv2x2(
    in_planes: int,
    out_planes: int,
    stride: int = 2,
    groups: int = 1,
    dilation: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=2,
        stride=stride,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class ReverseBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ReverseBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ReverseBasicBlock")

        self.conv1 = deconv2x2(inplanes, planes, stride) if stride == 2 else conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ReverseBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
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
        self.conv2 = (
            deconv2x2(width, width, stride, groups, dilation)
            if stride == 2
            else conv3x3(width, width, stride, groups, dilation)
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

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

        if self.upsample is not None:
            identity = self.upsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ReverseResNetDecoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        backbone_spec = resolve_backbone_spec(backbone_name)
        self._norm_layer = norm_layer
        self.groups = backbone_spec.groups
        self.base_width = backbone_spec.width_per_group
        self.dilation = 1
        self.inplanes = backbone_spec.bottleneck_channels

        block_cls: type[nn.Module] = ReverseBasicBlock if backbone_spec.expansion == 1 else ReverseBottleneck

        self.layer1 = self._make_layer(block_cls, 256, backbone_spec.decoder_layers[0], stride=2)
        self.layer2 = self._make_layer(block_cls, 128, backbone_spec.decoder_layers[1], stride=2)
        self.layer3 = self._make_layer(block_cls, 64, backbone_spec.decoder_layers[2], stride=2)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample_layers: list[nn.Module] = []
            if stride == 2:
                upsample_layers.append(deconv2x2(self.inplanes, planes * block.expansion, stride))
            else:
                upsample_layers.append(conv1x1(self.inplanes, planes * block.expansion))
            upsample_layers.append(norm_layer(planes * block.expansion))
            upsample = nn.Sequential(*upsample_layers)

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                upsample=upsample,
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

    def forward(self, x: Tensor) -> list[Tensor]:
        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)
        return [feature_c, feature_b, feature_a]


def build_decoder(backbone_name: str) -> ReverseResNetDecoder:
    return ReverseResNetDecoder(backbone_name=backbone_name)
