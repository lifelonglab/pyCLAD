from enum import Enum
from typing import Optional

from pydantic import BaseModel


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


class GRULayerConfig(LayerConfig):
    type: str = "GRU"
    input_size: int
    hidden_size: int
    activation: str
    dropout: float
    num_layers: int = 1
    batch_first: bool = True
    bidirectional: bool = False


class TCNLayerConfig(LayerConfig):
    type: str = "TCN"
    in_channels: int
    out_channels: int
    activation: Optional[str] = None
    dropout: Optional[float] = None
    kernel_size: int = 3
    dilation: int = 1
    padding: int = 1


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
