from typing import Tuple

import torch
from torch import nn


class ClipAdapter(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(width, bottleneck, bias=False), nn.LeakyReLU(inplace=False))
        self.fc2 = nn.Sequential(nn.Linear(bottleneck, width, bias=False), nn.LeakyReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.fc1(x)
        return bottleneck, self.fc2(bottleneck)
