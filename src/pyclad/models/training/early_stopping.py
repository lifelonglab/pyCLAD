from typing import Dict, Optional

from torch import Tensor, nn


class EarlyStopping:
    """Stateful early-stopping monitor.

    Tracks the best validation loss seen so far and, optionally, a snapshot of the
    module weights that produced it.

    An epoch counts as an improvement when ``val_loss < best_loss - min_delta``.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")

        self._patience = patience
        self._min_delta = min_delta
        self._restore_best_weights = restore_best_weights

        self.best_loss: float = float("inf")
        self._epochs_without_improvement = 0
        self._best_state: Optional[Dict[str, Tensor]] = None

    def step(self, val_loss: float, module: nn.Module) -> bool:
        """Record one epoch's validation loss. Return True when training should stop."""
        if val_loss < self.best_loss - self._min_delta:
            self.best_loss = val_loss
            self._epochs_without_improvement = 0
            if self._restore_best_weights:
                self._best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            return False

        self._epochs_without_improvement += 1
        return self._epochs_without_improvement > self._patience

    def restore(self, module: nn.Module) -> None:
        """Load the best-seen weights back into ``module`` (no-op if nothing was recorded)."""
        if self._best_state is not None:
            module.load_state_dict(self._best_state)

    def info(self) -> Dict[str, object]:
        return {
            "patience": self._patience,
            "min_delta": self._min_delta,
            "restore_best_weights": self._restore_best_weights,
        }
