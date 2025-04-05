from typing import Callable

import torch
import torch.nn as nn

from pyclad.models.autoencoder.builder import build_decoder, build_encoder
from pyclad.models.autoencoder.config import AutoencoderConfig
from pyclad.models.autoencoder.variational import VariationalDecoder, VariationalEncoder


class LSTMVariationalEncoder(VariationalEncoder):
    def __init__(self, config: AutoencoderConfig, builder: Callable = build_encoder) -> None:
        super(LSTMVariationalEncoder, self).__init__()
        self.seq_len = config.seq_len
        self.encoder: nn.ModuleList = builder(config.encoder)

        for layer in reversed(self.encoder):
            if isinstance(layer, nn.modules.LSTM):
                self.mean_layer = nn.Linear(layer.hidden_size, layer.hidden_size)
                self.logvar_layer = nn.Linear(layer.hidden_size, layer.hidden_size)
                break

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        hidden_state = hidden_state.reshape((batch_size, 1, hidden_size))

        mean = self.mean_layer(hidden_state)
        logvar = self.logvar_layer(hidden_state)

        return mean, logvar


class LSTMVariationalDecoder(VariationalDecoder):
    def __init__(self, config: AutoencoderConfig, builder: Callable = build_decoder) -> None:
        super(LSTMVariationalDecoder, self).__init__()
        self.seq_len = config.seq_len
        self.decoder: nn.ModuleList = builder(config.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        hidden_size, hidden_state = None, None
        x = x.repeat((1, self.seq_len, 1))

        for layer in self.decoder:
            if isinstance(layer, nn.LSTM):
                output, (_, _) = layer(x)
                hidden_size = layer.hidden_size
                x = output
            else:
                x = layer(x)

        return x.reshape((batch_size, self.seq_len, hidden_size))
