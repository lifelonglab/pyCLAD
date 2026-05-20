import copy
from typing import Optional, Union

import pytorch_lightning as pl
import torch


def resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available")
        if resolved_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(f"Requested device '{device}', but MPS is not available")
        return resolved_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def trainer_device_config(device: torch.device) -> tuple[str, int | list[int]]:
    if device.type == "cpu":
        return "cpu", 1
    if device.type == "cuda":
        if device.index is None:
            return "gpu", 1
        return "gpu", [device.index]
    if device.type == "mps":
        return "mps", 1
    return "auto", 1


def to_float(value: Union[None, float, int, torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class BestWeightsCallback(pl.Callback):
    def __init__(self, monitor: str, min_delta: float):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.best_state_dict: Optional[dict[str, torch.Tensor]] = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        current_loss = to_float(trainer.callback_metrics.get(self.monitor))
        if current_loss is None:
            return
        if self.best_loss is None or (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.best_state_dict = copy.deepcopy(pl_module.network.state_dict())
