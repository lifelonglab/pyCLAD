from typing import Sequence, Tuple

import torch
from torch.nn import functional as F

ANOMALY_CHANNEL = 1


def anomaly_scores_and_maps(
    patch_tokens: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    logit_scale: float,
    output_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = [logit_scale * F.normalize(tokens, dim=-1) @ text_features for tokens in patch_tokens]

    per_layer_scores = torch.stack([logit.softmax(dim=-1)[..., ANOMALY_CHANNEL].amax(dim=1) for logit in logits])
    scores = per_layer_scores.mean(dim=0)

    batch, length, channels = logits[0].shape
    grid = int(length**0.5)
    merged = torch.stack(logits, dim=1).mean(dim=1).permute(0, 2, 1).reshape(batch, channels, grid, grid)
    maps = F.interpolate(merged, size=output_size, mode="bilinear", align_corners=True)

    return scores, maps.softmax(dim=1)[:, ANOMALY_CHANNEL]
