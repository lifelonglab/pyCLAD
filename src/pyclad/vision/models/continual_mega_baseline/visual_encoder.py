from typing import List, Sequence, Tuple

import torch
from torch import nn

from pyclad.vision.models.continual_mega_baseline.adapter import ClipAdapter


class AdaptedVisualEncoder(nn.Module):
    def __init__(
        self,
        visual: nn.Module,
        feature_layers: Sequence[int],
        width: int,
        bottleneck: int,
        adapter_weight: float,
        noise_sigma: float,
        noise_in_float32: bool = False,
    ):
        super().__init__()
        self.visual = visual
        self.feature_layers = list(feature_layers)
        self.adapter_weight = adapter_weight
        self.noise_sigma = noise_sigma
        self.noise_in_float32 = noise_in_float32
        self.adapters = nn.ModuleList(ClipAdapter(width, bottleneck) for _ in self.feature_layers)

    def forward(self, images: torch.Tensor, with_noise: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = self._embed_patches(images)

        tokens: List[torch.Tensor] = []
        noisy_tokens: List[torch.Tensor] = []
        for index, block in enumerate(self.visual.transformer.resblocks):
            x = block(x, attn_mask=None)
            if index + 1 not in self.feature_layers:
                continue

            adapter = self.adapters[self.feature_layers.index(index + 1)]
            bottleneck, adapted = adapter(x)
            if with_noise:
                dtype = torch.float32 if self.noise_in_float32 else x.dtype
                noise = torch.normal(0.0, self.noise_sigma, x.shape, device=x.device, dtype=dtype)
                noisy_bottleneck, _ = adapter(x + noise)
                noisy_tokens.append(_patch_tokens(noisy_bottleneck))

            x = (1.0 - self.adapter_weight) * x + self.adapter_weight * adapted
            tokens.append(_patch_tokens(bottleneck))

        return tokens, noisy_tokens

    def _embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        x = self.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = self.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(self.visual.patch_dropout(x))
        return x.permute(1, 0, 2)


def _patch_tokens(sequence: torch.Tensor) -> torch.Tensor:
    return sequence.permute(1, 0, 2)[:, 1:, :]
