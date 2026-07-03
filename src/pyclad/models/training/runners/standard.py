import logging
from typing import Optional

from pyclad.models.training.early_stopping import EarlyStopping
from pyclad.models.training.runners.runner import RunResult, TorchRunner

logger = logging.getLogger(__name__)


class StandardRunner(TorchRunner):
    """
    The training loop, with validation and early stopping as optional add-ons.
    """
    def __init__(
        self,
        max_epochs: int,
        validation_fraction: float = 0.0,
        early_stopping: Optional[EarlyStopping] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(validation_fraction=validation_fraction, seed=seed)
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if early_stopping is not None and validation_fraction <= 0.0:
            raise ValueError(
                f"early stopping needs a validation set to monitor; "
                f"validation_fraction must be in (0, 1), got {validation_fraction}"
            )

        self._max_epochs = max_epochs
        self._early_stopping = early_stopping

    def run(self, backbone, train_loader, loss_fn, *, grad_callback=None, val_loader=None, val_loss_fn=None):
        module = backbone.get_module()
        optimizer = backbone.get_optimizer()

        can_validate = val_loader is not None and val_loss_fn is not None
        if self._early_stopping is not None and not can_validate:
            logger.warning(
                "Early stopping is configured but no validation data was passed to run(); "
                "training for the full %d epochs instead.",
                self._max_epochs,
            )

        epochs_run = 0
        stopped_early = False
        final_val_loss = None
        for _ in range(self._max_epochs):
            self._train_epoch(module, optimizer, train_loader, loss_fn, grad_callback)
            epochs_run += 1
            if can_validate:
                final_val_loss = self._validate(module, val_loader, val_loss_fn)
                if self._early_stopping is not None and self._early_stopping.step(final_val_loss, module):
                    stopped_early = True
                    break

        best_val_loss = None
        if self._early_stopping is not None and can_validate:
            self._early_stopping.restore(module)
            best_val_loss = self._early_stopping.best_loss

        result = RunResult(epochs_run, stopped_early, best_val_loss, final_val_loss)
        logger.debug("StandardRunner finished: %s", result)
        return result

    def name(self) -> str:
        return "StandardRunner"

    def additional_info(self) -> dict:
        return {
            "max_epochs": self._max_epochs,
            "validation_fraction": self.validation_fraction,
            "early_stopping": self._early_stopping.info() if self._early_stopping is not None else None,
        }
