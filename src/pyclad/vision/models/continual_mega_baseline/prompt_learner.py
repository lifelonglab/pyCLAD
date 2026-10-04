from typing import Callable, List, Mapping, Sequence

import torch
from torch import nn

from pyclad.vision.models.continual_mega_baseline.text_encoder import PromptGroup


class PromptLearner(nn.Module):
    def __init__(
        self,
        token_embedding: nn.Module,
        context_dim: int,
        prompt_groups: Mapping[str, Sequence[str]],
        n_ctx: int,
        tokenize: Callable[[Sequence[str]], torch.Tensor],
    ):
        super().__init__()
        self.groups = list(prompt_groups)
        learnable_prefix = " ".join(["X"] * n_ctx)
        context = {}

        for group, prompts in prompt_groups.items():
            tokenized = tokenize([f"{learnable_prefix} {prompt}." for prompt in prompts])
            with torch.no_grad():
                embedded = token_embedding(tokenized)

            self.register_buffer(f"{group}_tokenized", tokenized, persistent=False)
            self.register_buffer(f"{group}_prefix", embedded[:, :1, :], persistent=False)
            self.register_buffer(f"{group}_suffix", embedded[:, 1 + n_ctx :, :], persistent=False)

            vectors = torch.empty(len(prompts), n_ctx, context_dim)
            nn.init.normal_(vectors, std=0.02)
            context[f"{group}_end"] = nn.Parameter(vectors)

        self.ctx = nn.ParameterDict(context)

    def forward(self) -> List[PromptGroup]:
        groups = []
        for group in self.groups:
            prompts = torch.cat(
                [
                    getattr(self, f"{group}_prefix"),
                    self.ctx[f"{group}_end"],
                    getattr(self, f"{group}_suffix"),
                ],
                dim=1,
            )
            groups.append((prompts, getattr(self, f"{group}_tokenized")))
        return groups
