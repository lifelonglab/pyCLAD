from __future__ import annotations

import math

import torch
from torch import nn


def _global_affine_init_value(global_affine_init: float = 1.0) -> float:
    return 2.0 * math.log(math.exp(5.0 * global_affine_init) - 1.0)


class ConvSubnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_ratio: float, kernel_size: int):
        super().__init__()
        hidden_channels = max(1, int(in_channels * hidden_ratio))
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FastFlowStep(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_ratio: float,
        kernel_size: int,
        affine_clamping: float,
    ):
        super().__init__()

        self.split_len1 = channels - channels // 2
        self.split_len2 = channels // 2
        self.affine_clamping = affine_clamping
        self.softplus = nn.Softplus(beta=0.5)

        self.register_buffer("permutation", torch.randperm(channels))

        global_scale_value = _global_affine_init_value()
        self.global_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(global_scale_value)))
        self.global_offset = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.subnet = ConvSubnet(
            in_channels=self.split_len1,
            out_channels=2 * self.split_len2,
            hidden_ratio=hidden_ratio,
            kernel_size=kernel_size,
        )

    def _global_scale_activation(self) -> torch.Tensor:
        return 0.1 * self.softplus(self.global_scale)

    def _affine(self, x: torch.Tensor, affine_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        affine_params = affine_params * 0.1
        log_scale = self.affine_clamping * torch.tanh(affine_params[:, : self.split_len2])
        shift = affine_params[:, self.split_len2 :]
        transformed = x * torch.exp(log_scale) + shift
        log_det = torch.sum(log_scale, dim=(1, 2, 3))
        return transformed, log_det

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], dim=1)
        affine_params = self.subnet(x1)
        x2, log_det = self._affine(x2, affine_params)
        output = torch.cat((x1, x2), dim=1)

        global_scale = self._global_scale_activation()
        output = output * global_scale + self.global_offset
        output = output[:, self.permutation, :, :]

        pixels_per_channel = output.shape[-2] * output.shape[-1]
        global_log_det = pixels_per_channel * torch.sum(torch.log(global_scale))
        log_det = log_det + global_log_det
        return output, log_det


class FastFlowSequence(nn.Module):
    def __init__(
        self,
        channels: int,
        flow_steps: int,
        conv3x3_only: bool,
        hidden_ratio: float,
        affine_clamping: float,
    ):
        super().__init__()
        steps: list[nn.Module] = []
        for index in range(flow_steps):
            kernel_size = 3 if conv3x3_only or index % 2 == 0 else 1
            steps.append(
                FastFlowStep(
                    channels=channels,
                    hidden_ratio=hidden_ratio,
                    kernel_size=kernel_size,
                    affine_clamping=affine_clamping,
                )
            )
        self.steps = nn.ModuleList(steps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_log_det = x.new_zeros((x.shape[0],))
        hidden = x
        for step in self.steps:
            hidden, log_det = step(hidden)
            total_log_det = total_log_det + log_det
        return hidden, total_log_det
