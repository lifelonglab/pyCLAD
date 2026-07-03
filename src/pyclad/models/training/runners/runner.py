import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.output.output_writer import InfoProvider

LossFn = Callable[[Any], Tensor]
GradCallback = Callable[[nn.Module], None]


@dataclass
class RunResult:
    """Outcome of a single training run, for logging and inspection.

    ``final_val_loss`` is the last epoch's validation loss whenever a validation set was
    evaluated (with or without early stopping). ``best_val_loss`` is the best value an
    early-stopping monitor saw, and is None when no monitor was used.
    """

    epochs_run: int
    stopped_early: bool = False
    best_val_loss: Optional[float] = None
    final_val_loss: Optional[float] = None


class TorchRunner(InfoProvider, abc.ABC):
    """Owns the training loop for a TorchBackbone.

    Separates *how training runs* (epochs, validation, early stopping) from *what is
    trained* (the backbone) and *what loss is optimized* (the strategy's ``loss_fn``).

    :param validation_fraction: fraction of each concept's data to hold out for
        validation. 0.0 (the default) means the runner does not use a validation set.
    :param seed: makes the validation split deterministic across concepts within a run.
    """

    def __init__(self, validation_fraction: float = 0.0, seed: Optional[int] = None) -> None:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction}")
        self.validation_fraction = validation_fraction
        self._seed = seed

    def split_train_test(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split concept data into (train, validation)
        """
        if self.validation_fraction <= 0.0:
            return data, None
        train, val = train_test_split(
            data, test_size=self.validation_fraction, random_state=self._seed, shuffle=True
        )
        return train, val

    @abc.abstractmethod
    def run(
        self,
        backbone: TorchBackbone,
        train_loader: DataLoader,
        loss_fn: LossFn,
        *,
        grad_callback: Optional[GradCallback] = None,
        val_loader: Optional[DataLoader] = None,
        val_loss_fn: Optional[LossFn] = None,
    ) -> RunResult:
        """Train ``backbone`` on ``train_loader`` minimizing ``loss_fn`` per batch."""
        ...

    def info(self):
        return {"name": self.name(), **self.additional_info()}

    @abc.abstractmethod
    def name(self) -> str: ...

    def additional_info(self) -> dict:
        return {}

    # -- shared training helpers -------------------------------------------------

    @staticmethod
    def _train_epoch(
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        loss_fn: LossFn,
        grad_callback: Optional[GradCallback],
    ) -> None:
        was_training = module.training
        module.train()
        try:
            for batch in train_loader:
                loss = loss_fn(batch)
                optimizer.zero_grad()
                loss.backward()
                if grad_callback is not None:
                    grad_callback(module)
                optimizer.step()
        finally:
            module.train(was_training)

    @staticmethod
    def _validate(module: nn.Module, val_loader: DataLoader, val_loss_fn: LossFn) -> float:
        was_training = module.training
        module.eval()
        try:
            total_loss, total_samples = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    n = len(batch[0])
                    total_loss += float(val_loss_fn(batch)) * n
                    total_samples += n
            return total_loss / max(total_samples, 1)
        finally:
            module.train(was_training)
