"""Training and replay policy for the ContTrain++ strategy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContTrainPlusPlusConfig:
    """Continual optimization, replay, scheduling, and shuffle settings."""

    epochs: int = 100
    batch_size: int = 32
    replay_batch_size: int = 16
    buffer_size: int = 1_000
    gradient_clip: float = 1.0
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    novelty_mad_scale: float = 3.0
    novelty_warmup_epochs: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "buffer_size", "lr_step_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.replay_batch_size < self.batch_size:
            raise ValueError("replay_batch_size must be in [0, batch_size)")
        if self.gradient_clip <= 0 or self.lr_gamma <= 0:
            raise ValueError("gradient_clip and lr_gamma must be positive")
        if self.novelty_mad_scale < 0 or self.novelty_warmup_epochs < 0:
            raise ValueError("novelty controls must be non-negative")
