"""Architecture, loss, and training configuration for COMMAND."""

from dataclasses import asdict, dataclass, field
from math import isfinite


@dataclass(frozen=True)
class CommandArchitectureConfig:
    """Fixed dimensions for PyCLAD's COMMAND paper recreation."""

    appearance_dim: int = 1024
    motion_dim: int = 1024
    hidden_dim: int = 2048
    state_dim: int = 32
    temporal_kernel_size: int = 5
    mamba_blocks: int = 1
    memory_size: int = 256
    projection_dim: int = 128
    cbam_reduction: int = 8
    cbam_temporal_kernel_size: int = 7
    dropout: float = 0.25

    def __post_init__(self) -> None:
        integer_fields = {
            "appearance_dim": self.appearance_dim,
            "motion_dim": self.motion_dim,
            "hidden_dim": self.hidden_dim,
            "state_dim": self.state_dim,
            "temporal_kernel_size": self.temporal_kernel_size,
            "mamba_blocks": self.mamba_blocks,
            "memory_size": self.memory_size,
            "projection_dim": self.projection_dim,
            "cbam_reduction": self.cbam_reduction,
            "cbam_temporal_kernel_size": self.cbam_temporal_kernel_size,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd")
        if self.cbam_temporal_kernel_size % 2 == 0:
            raise ValueError("cbam_temporal_kernel_size must be odd")
        if self.fused_dim < self.cbam_reduction:
            raise ValueError("fused_dim must be at least cbam_reduction")

    @property
    def fused_dim(self) -> int:
        return self.appearance_dim + self.motion_dim


@dataclass(frozen=True)
class CommandLossConfig:
    """Published composite weights plus explicitly documented paper-silent defaults."""

    mil_weight: float = 1.0
    contrastive_weight: float = 1.0
    focal_weight: float = 0.5
    anomaly_separation_weight: float = 0.3
    contrastive_temperature: float = 0.15
    mil_margin: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.5
    anomaly_margin: float = 1.0
    sparsity_weight: float = 8e-5
    smoothness_weight: float = 8e-5
    score_l2_weight: float = 5e-5

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if name == "focal_alpha":
                if not 0.0 <= value <= 1.0:
                    raise ValueError("focal_alpha must be in [0, 1]")
            elif value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.contrastive_temperature == 0.0:
            raise ValueError("contrastive_temperature must be positive")


@dataclass(frozen=True)
class CommandTrainerConfig:
    """Optimization, replay, and reproducibility settings for ContTrain++.

    ``threshold`` overrides the video decision threshold in raw dual-memory
    deviation units. Otherwise, fit calibrates ``threshold_quantile`` of the
    normal videos' maximum scores, separately from snippet score normalization.
    """

    architecture: CommandArchitectureConfig = field(default_factory=CommandArchitectureConfig)
    loss: CommandLossConfig = field(default_factory=CommandLossConfig)
    epochs: int = 100
    batch_size: int = 32
    replay_batch_size: int = 16
    buffer_size: int = 1_000
    learning_rate: float = 1e-4
    secondary_memory_learning_rate: float = 1e-5
    gradient_clip: float = 1.0
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    novelty_mad_scale: float = 3.0
    novelty_warmup_epochs: int = 1
    seed: int = 42
    device: str = "cpu"
    threshold: float | None = None
    threshold_quantile: float = 0.99

    def __post_init__(self) -> None:
        integer_fields = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "lr_step_size": self.lr_step_size,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.replay_batch_size < self.batch_size:
            raise ValueError("replay_batch_size must be in [0, batch_size)")
        if self.learning_rate <= 0 or self.secondary_memory_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.gradient_clip <= 0 or self.lr_gamma <= 0:
            raise ValueError("gradient_clip and lr_gamma must be positive")
        if self.novelty_mad_scale < 0 or self.novelty_warmup_epochs < 0:
            raise ValueError("novelty controls must be non-negative")
        if self.threshold is not None and (not isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("threshold must be finite and non-negative")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
