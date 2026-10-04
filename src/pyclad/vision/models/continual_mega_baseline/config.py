from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class ContinualMegaBaselineConfig:
    weights_path: Union[str, Path]
    clip_model_name: str = "ViT-L-14-336"
    image_size: int = 336
    feature_layers: Tuple[int, ...] = (6, 12, 18, 24)
    bottleneck: int = 768
    width: int = 1024
    n_ctx: int = 8
    adapter_weight: float = 0.1
    logit_scale: float = 100.0
    noise_sigma: float = 0.25
    noise_in_float32: bool = False
    batch_size: int = 16
    train_batch_size: int = 16
    epochs: int = 50
    train_prompts: bool = True
    use_synthetic_anomalies: bool = True
    learning_rate: float = 1e-4
    prompt_learning_rate: float = 1e-4
    seed: int = 2025
    device: Optional[str] = None
