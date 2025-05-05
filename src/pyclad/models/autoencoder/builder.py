import inspect

import torch.nn as nn

from pyclad.models.autoencoder.config import (
    DecoderConfig,
    EncoderConfig,
)


def _build_lstm(config: EncoderConfig | DecoderConfig) -> nn.ModuleList:
    layers = []
    input_size = config.layers[0].input_size
    for layer_config in config.layers:
        if layer_type := layer_config.type != "LSTM":
            raise ValueError(f"Unsupported layer type: {layer_type}")

        layer_class = nn.LSTM
        layer_params = inspect.signature(layer_class.__base__.__init__).parameters.keys() - {"self"}
        layer_kwargs = {k: v for k, v in layer_config.dict().items() if k in layer_params}
        layer_kwargs["input_size"] = input_size

        layers.append(layer_class(**layer_kwargs))

        if layer_config.activation:
            layers.append(getattr(nn, layer_config.activation)())
        if layer_config.dropout:
            layers.append(nn.Dropout(layer_config.dropout))
        input_size = layer_config.hidden_size * (2 if layer_config.bidirectional else 1)

    return nn.ModuleList(layers)


def _build_gru(config: EncoderConfig | DecoderConfig) -> nn.ModuleList:
    layers = []
    input_size = config.layers[0].input_size
    for layer_config in config.layers:
        if layer_type := layer_config.type != "GRU":
            raise ValueError(f"Unsupported layer type: {layer_type}")

        layer_class = nn.GRU
        layer_params = inspect.signature(layer_class.__base__.__init__).parameters.keys() - {"self"}
        layer_kwargs = {k: v for k, v in layer_config.dict().items() if k in layer_params}
        layer_kwargs["input_size"] = input_size

        layers.append(layer_class(**layer_kwargs))

        if layer_config.activation:
            layers.append(getattr(nn, layer_config.activation)())
        if layer_config.dropout:
            layers.append(nn.Dropout(layer_config.dropout))
        input_size = layer_config.hidden_size * (2 if layer_config.bidirectional else 1)

    return nn.ModuleList(layers)


def _build_tcn(config: EncoderConfig | DecoderConfig) -> nn.ModuleList:
    layers = []
    in_channels = config.layers[0].in_channels
    for layer_config in config.layers:
        if layer_type := layer_config.type != "TCN":
            raise ValueError(f"Unsupported layer type: {layer_type}")

        layer_class = nn.Conv1d
        layer_params = inspect.signature(layer_class.__base__.__init__).parameters.keys() - {"self"}
        layer_kwargs = {k: v for k, v in layer_config.dict().items() if k in layer_params}
        layer_kwargs["in_channels"] = in_channels

        layers.append(layer_class(**layer_kwargs))

        if layer_config.activation:
            layers.append(getattr(nn, layer_config.activation)())
        if layer_config.dropout:
            layers.append(nn.Dropout(layer_config.dropout))
        in_channels = layer_config.out_channels

    return nn.ModuleList(layers)


def build_lstm_encoder(encoder_config: EncoderConfig) -> nn.ModuleList:
    return _build_lstm(encoder_config)


def build_lstm_decoder(decoder_config: DecoderConfig) -> nn.ModuleList:
    return _build_lstm(decoder_config)


def build_gru_encoder(encoder_config: EncoderConfig) -> nn.ModuleList:
    return _build_gru(encoder_config)


def build_gru_decoder(decoder_config: DecoderConfig) -> nn.ModuleList:
    return _build_gru(decoder_config)


def build_tcn_encoder(encoder_config: EncoderConfig) -> nn.ModuleList:
    return _build_tcn(encoder_config)


def build_tcn_decoder(decoder_config: DecoderConfig) -> nn.ModuleList:
    return _build_tcn(decoder_config)
