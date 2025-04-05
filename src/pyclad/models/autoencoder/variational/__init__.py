from functools import cached_property

import torch.nn as nn


class VariationalEncoder(nn.Module):
    def __init__(self) -> None:
        super(VariationalEncoder, self).__init__()
        self._is_variational: bool = True

    @cached_property
    def is_variational(self):
        return self._is_variational


class VariationalDecoder(nn.Module):
    def __init__(self) -> None:
        super(VariationalDecoder, self).__init__()
        self._is_variational: bool = True

    @cached_property
    def is_variational(self):
        return self._is_variational
