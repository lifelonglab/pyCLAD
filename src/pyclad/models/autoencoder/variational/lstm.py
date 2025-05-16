import torch
import torch.nn as nn


class LSTMVariationalEncoder(nn.Module):
    def __init__(self, encoder: nn.ModuleList, seq_len: int) -> None:
        super(LSTMVariationalEncoder, self).__init__()
        self.encoder: nn.ModuleList = encoder
        self.seq_len = seq_len

        for layer in reversed(self.encoder):
            if isinstance(layer, nn.modules.LSTM):
                self.mean_layer = nn.Linear(layer.hidden_size, layer.hidden_size)
                self.logvar_layer = nn.Linear(layer.hidden_size, layer.hidden_size)
                break

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
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


class LSTMVariationalDecoder(nn.Module):
    def __init__(self, decoder: nn.ModuleList, seq_len: int) -> None:
        super(LSTMVariationalDecoder, self).__init__()
        self.decoder: nn.ModuleList = decoder
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
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
