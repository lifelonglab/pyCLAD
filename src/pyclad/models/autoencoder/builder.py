import torch.nn as nn

from pyclad.models.autoencoder.config import AutoencoderConfig, LayerConfig


def _build_layer(layer_config: LayerConfig) -> nn.Module:
    return layer_config.cls(**layer_config.kwargs)


def build(config: AutoencoderConfig) -> tuple[nn.ModuleList, nn.ModuleList]:
    encoder: list[nn.Module] = []
    decoder: list[nn.Module] = []

    for layer_config in config.encoder.layers:
        encoder.append(_build_layer(layer_config))

    for layer_config in config.decoder.layers:
        decoder.append(_build_layer(layer_config))

    return nn.ModuleList(encoder), nn.ModuleList(decoder)
