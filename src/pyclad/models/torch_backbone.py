import abc
from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from pyclad.output.output_writer import InfoProvider


class TorchBackbone(InfoProvider, abc.ABC):
    """PyTorch model component used by continual learning strategies.

    Separates model internals (architecture, loss, optimizer) from strategies.
    Strategies depend on this interface rather than on concrete models, so the same strategy works with any compliant backbone.
    """

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Return the underlying PyTorch module."""
        ...

    @abc.abstractmethod
    def get_optimizer(self) -> Optimizer:
        """Return a fresh optimizer over the current module parameters."""
        ...

    @abc.abstractmethod
    def compute_loss(self, x: Tensor) -> Tensor:
        """Return the base task loss as a scalar with a grad_fn attached.
        Must be called outside any no_grad context so strategies can call backward() on the result.
        """
        ...

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Return the primary output tensor (e.g. reconstruction for autoencoders).
        Unlike compute_loss, this is the raw output without reduction — safe to call inside no_grad for distillation targets.
        """
        ...

    def to(self, device: torch.device) -> "TorchBackbone":
        self.get_module().to(device)
        return self

    def device(self) -> torch.device:
        return next(self.get_module().parameters()).device

    def info(self) -> Dict[str, Any]:
        return {"model": {"name": self.name(), **self.additional_info()}}

    def additional_info(self):
        return {}

    @abc.abstractmethod
    def name(self):
        ...
