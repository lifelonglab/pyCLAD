from typing import Callable

import torch
import torch.nn as nn

from pyclad.models.autoencoder.builder import build_tcn_decoder, build_tcn_encoder
from pyclad.models.autoencoder.config import AutoencoderConfig
from pyclad.models.autoencoder.standard import Decoder, Encoder


class TCNEncoder(Encoder):
    def __init__(self, config: AutoencoderConfig, builder: Callable = build_tcn_encoder) -> None:
        super(TCNEncoder, self).__init__()
        self.seq_len = config.seq_len
        self.encoder: nn.ModuleList = builder(config.encoder)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, in_channels) -> (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)

        for layer in self.encoder:
            x = layer(x)

        x = self.pool(x)

        # (batch_size, out_channels, 1)
        return x


class TCNDecoder(Decoder):
    def __init__(self, config: AutoencoderConfig, builder: Callable = build_tcn_decoder) -> None:
        super(TCNDecoder, self).__init__()
        self.seq_len = config.seq_len

        self.decoder: nn.ModuleList = builder(config.decoder)

        for layer in self.decoder:
            if isinstance(layer, nn.modules.Conv1d):
                self.linear = nn.Linear(layer.in_channels, layer.in_channels * self.seq_len)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, out_channels, 1) -> (batch_size, out_channels)
        x = x.squeeze(-1)
        batch_size, out_channels = x.shape

        # (batch_size, out_channels) -> (batch_size, out_channels * seq_len)
        x = self.linear(x)

        # (batch_size, out_channels * seq_len) -> (batch_size, out_channels, seq_len)
        x = x.view(-1, out_channels, self.seq_len)

        for layer in self.decoder:
            x = layer(x)

        # (batch_size, out_channels, seq_len) -> (batch_size, seq_len, out_channels)
        return x.permute(0, 2, 1)
