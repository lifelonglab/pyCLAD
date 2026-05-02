import abc
from abc import abstractmethod
from typing import TypedDict

import torch
from torch import nn

from pyclad.models.model import Model


class TrainingStepOutput(TypedDict):
    """Output for each training step.

    Useful for buffer-based methods.
    It is compatible with LightningModule.training_step.
    """

    loss: torch.Tensor
    input: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor


class TorchBackedModel(Model, abc.ABC):
    """Interface for a PyTorch model, exposing it's trainign module and loss.

    It is useful for regularization-based methods. In their case training
    becomes responsibility of the strategy which uses some model.

    See examples/derpp_example.py to see how such model can be used with DER++
    regularization.
    """

    @property
    @abstractmethod
    def trainable_module(self) -> nn.Module:
        """Exposes trainable torch module."""
        ...

    @property
    @abstractmethod
    def loss_function(self) -> nn.Module:
        """Base loss function for a given module."""
