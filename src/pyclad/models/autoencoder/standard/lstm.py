from typing import Callable

import torch
import torch.nn as nn

from pyclad.models.autoencoder.builder import build_lstm_decoder, build_lstm_encoder
from pyclad.models.autoencoder.config import AutoencoderConfig


class LSTMEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, builder: Callable = build_lstm_encoder) -> None:
        super(LSTMEncoder, self).__init__()
        self.seq_len = config.seq_len
        self.encoder: nn.ModuleList = builder(config.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        hidden_size, hidden_state = None, None
        num_layers = None

        for layer in self.encoder:
            if isinstance(layer, nn.LSTM):
                output, (hidden_state, cell_state) = layer(x)
                hidden_size = layer.hidden_size
                num_layers = layer.num_layers
                x = output
            else:
                x = layer(x)

        if num_layers > 1:
            hidden_size = hidden_state[-1]

        return hidden_state.reshape((batch_size, 1, hidden_size))


class LSTMDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, builder: Callable = build_lstm_decoder) -> None:
        super(LSTMDecoder, self).__init__()
        self.seq_len = config.seq_len
        self.decoder: nn.ModuleList = builder(config.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        hidden_size, hidden_state = None, None
        x = x.repeat((1, self.seq_len, 1))

        for layer in self.decoder:
            if isinstance(layer, nn.LSTM):
                output, (hidden_state, cell_state) = layer(x)
                hidden_size = layer.hidden_size
                x = output
            else:
                x = layer(x)

        return x.reshape((batch_size, self.seq_len, hidden_size))
