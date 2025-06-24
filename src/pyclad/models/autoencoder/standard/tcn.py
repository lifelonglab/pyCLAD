import torch
import torch.nn as nn


class TCNEncoder(nn.Module):
    def __init__(self, encoder: nn.ModuleList, seq_len: int) -> None:
        super(TCNEncoder, self).__init__()
        self.encoder: nn.ModuleList = encoder
        self.seq_len = seq_len

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, in_channels) -> (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)

        for layer in self.encoder:
            x = layer(x)

        x = self.pool(x)

        # (batch_size, out_channels, 1)
        return x


class TCNDecoder(nn.Module):
    def __init__(self, decoder: nn.ModuleList, seq_len: int) -> None:
        super(TCNDecoder, self).__init__()
        self.seq_len = seq_len

        for layer in decoder:
            if isinstance(layer, nn.modules.ConvTranspose1d):
                self.linear = nn.Linear(layer.in_channels, layer.in_channels * self.seq_len)
                break

        self.decoder: nn.ModuleList = decoder

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
