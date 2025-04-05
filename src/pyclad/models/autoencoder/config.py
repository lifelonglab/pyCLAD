from enum import Enum
from typing import Type

import torch.nn as nn
from pydantic import BaseModel


def get_layer_class(layer_type: str) -> Type[nn.Module]:
    if layer_type == "LSTM":
        return nn.LSTM
    raise ValueError(f"Unsupported layer type: {layer_type}")


class AutoencoderType(str, Enum):
    STANDARD = "standard"
    VARIATIONAL = "variational"


class LayerConfig(BaseModel):
    type: str


class LSTMLayerConfig(LayerConfig):
    type: str = "LSTM"
    input_size: int
    hidden_size: int
    activation: str
    dropout: float
    num_layers: int = 1
    batch_first: bool = True
    bidirectional: bool = False


class EncoderConfig(BaseModel):
    layers: list[LayerConfig]


class DecoderConfig(BaseModel):
    layers: list[LayerConfig]


class AutoencoderConfig(BaseModel):
    seq_len: int
    encoder: EncoderConfig
    decoder: DecoderConfig


class StandardAutoencoderConfig(AutoencoderConfig):
    type: AutoencoderType = AutoencoderType.STANDARD


class VariationalAutoencoderConfig(AutoencoderConfig):
    type: AutoencoderType = AutoencoderType.VARIATIONAL
