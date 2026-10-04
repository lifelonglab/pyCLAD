from pathlib import Path
from typing import Union

import torch
from torch import nn


def load_reference_checkpoint(path: Union[str, Path], adapters: nn.Module, prompt_learner: nn.Module) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    adapters.load_state_dict(checkpoint["adapters"])
    prompt_learner.load_state_dict(checkpoint["prompt_state_dict"])
