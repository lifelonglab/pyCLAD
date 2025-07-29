import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, encoder: nn.ModuleList) -> None:
        super(GRUEncoder, self).__init__()
        self.encoder: nn.ModuleList = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        hidden_size, hidden_state = None, None
        num_layers = None

        for layer in self.encoder:
            if isinstance(layer, nn.GRU):
                output, hidden_state = layer(x)
                hidden_size = layer.hidden_size
                num_layers = layer.num_layers
                x = output
            else:
                x = layer(x)

        if num_layers > 1:
            hidden_size = hidden_state[-1]

        return hidden_state.reshape((batch_size, 1, hidden_size))


class GRUDecoder(nn.Module):
    def __init__(self, decoder: nn.ModuleList, seq_len: int) -> None:
        super(GRUDecoder, self).__init__()
        self.decoder: nn.ModuleList = decoder
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        hidden_size, hidden_state = None, None
        x = x.repeat((1, self.seq_len, 1))

        for layer in self.decoder:
            if isinstance(layer, nn.GRU):
                output, hidden_state = layer(x)
                hidden_size = layer.hidden_size
                x = output
            else:
                x = layer(x)

        return x.reshape((batch_size, self.seq_len, hidden_size))
