from typing import Literal, Optional

from pydantic import BaseModel, Field


class VisionConfig(BaseModel):
    """Configuration shared by all vision anomaly-detection models.

    Holds only fields used for preprocessing, scoring and threshold calibration,
    i.e. everything that is independent of how the model is trained.
    """

    input_size: tuple[int, int] = (224, 224)
    batch_size: int = Field(default=32, gt=0)

    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    normalize_mean: Optional[tuple[float, ...]] = (0.485, 0.456, 0.406)
    normalize_std: Optional[tuple[float, ...]] = (0.229, 0.224, 0.225)
    input_range: Literal["uint8", "float01"] = "uint8"
    input_layout: Literal["NHWC", "NCHW"] = "NHWC"

    device: Optional[str] = None

    # Reproducibility. ``seed=None`` (default) leaves global RNG state untouched. When set,
    # data shuffling is made reproducible via a local DataLoader generator, and models seed
    # weight init / channel permutations through ``_apply_seed()``. Note: strict bitwise
    # determinism (e.g. ``torch.use_deterministic_algorithms``) is intentionally NOT handled
    # here — it is a process-wide, non-reversible setting that belongs at the experiment
    # entry point, not in a model constructor.
    seed: Optional[int] = None


class LightningVisionConfig(VisionConfig):
    """Adds gradient-training fields for models trained via a PyTorch Lightning trainer."""

    epochs: int = Field(default=100, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True
