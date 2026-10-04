from pathlib import Path
from typing import Union

import open_clip
from torch import nn

from pyclad.vision.models.continual_mega_baseline.attention import (
    use_reference_attention,
)

OPENAI_STATE_DICT_KEYS_TO_DROP = ("input_resolution", "context_length", "vocab_size")


def build_clip_backbone(model_name: str, weights_path: Union[str, Path]) -> nn.Module:
    pretrained = open_clip.load_openai_model(str(weights_path), precision="fp32", device="cpu")
    state_dict = pretrained.state_dict()
    for key in OPENAI_STATE_DICT_KEYS_TO_DROP:
        state_dict.pop(key, None)

    model = open_clip.create_model(model_name)
    model.load_state_dict(state_dict, strict=True)
    _require_sequence_first(model.visual.transformer)
    use_reference_attention(model.visual.transformer)
    use_reference_attention(model.transformer)
    return model.eval().requires_grad_(False)


def _require_sequence_first(transformer: nn.Module) -> None:
    if getattr(transformer, "batch_first", False):
        raise RuntimeError(
            "open_clip uses a batch-first transformer, but this code feeds it sequence-first tensors. "
            "Requires open_clip_torch < 3.0."
        )
