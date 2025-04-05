import torch.nn as nn

from pyclad.models.autoencoder.config import (
    DecoderConfig,
    EncoderConfig,
    get_layer_class,
)


def _build(config: EncoderConfig | DecoderConfig) -> nn.ModuleList:
    layers = []
    input_size = config.layers[0].input_size
    for layer_config in config.layers:
        layer_class = get_layer_class(layer_config.type)
        layers.append(
            layer_class(
                input_size=input_size,
                hidden_size=layer_config.hidden_size,
                num_layers=layer_config.num_layers,
                batch_first=layer_config.batch_first,
                bidirectional=layer_config.bidirectional,
            )
        )
        if layer_config.activation:
            layers.append(getattr(nn, layer_config.activation)())
        if layer_config.dropout:
            layers.append(nn.Dropout(layer_config.dropout))
        input_size = layer_config.hidden_size * (2 if layer_config.bidirectional else 1)

    return nn.ModuleList(layers)


def build_encoder(encoder_config: EncoderConfig) -> nn.ModuleList:
    return _build(encoder_config)


def build_decoder(decoder_config: DecoderConfig) -> nn.ModuleList:
    return _build(decoder_config)
