from typing import Any, Optional, Type

import torch.nn as nn
from pydantic import BaseModel


class LayerConfig(BaseModel): ...


class ActivationLayerConfig(LayerConfig):
    cls: Type[nn.Module]
    kwargs: Optional[dict[str, Any]] = {}


class DropoutLayerConfig(LayerConfig):
    cls: Type[nn.Module] = nn.Dropout
    kwargs: dict[str, Any] = {}


class LSTMLayerConfig(LayerConfig):
    cls: Type[nn.Module] = nn.LSTM
    kwargs: dict[str, Any] = {}


class GRULayerConfig(LayerConfig):
    cls: Type[nn.Module] = nn.GRU
    kwargs: dict[str, Any] = {}


class Conv1dLayerConfig(LayerConfig):
    cls: Type[nn.Module] = nn.Conv1d
    kwargs: dict[str, Any] = {}


class ConvTranspose1dLayerConfig(LayerConfig):
    cls: Type[nn.Module] = nn.ConvTranspose1d
    kwargs: dict[str, Any] = {}


class EncoderConfig(BaseModel):
    layers: list[LayerConfig]


class DecoderConfig(BaseModel):
    layers: list[LayerConfig]


class AutoencoderConfig(BaseModel):
    seq_len: int
    encoder: EncoderConfig
    decoder: DecoderConfig
