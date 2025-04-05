from functools import cached_property

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self._is_variational: bool = False

    @cached_property
    def is_variational(self):
        return self._is_variational


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self._is_variational: bool = False

    @cached_property
    def is_variational(self):
        return self._is_variational
