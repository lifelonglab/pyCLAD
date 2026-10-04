from typing import Sequence, Tuple

import torch
from torch import nn

PromptGroup = Tuple[torch.Tensor, torch.Tensor]


class ClipTextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompt_embeddings + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        end_of_text = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0]), end_of_text] @ self.text_projection


def encode_prompt_groups(encoder: ClipTextEncoder, groups: Sequence[PromptGroup]) -> torch.Tensor:
    features = []
    for prompt_embeddings, tokenized_prompts in groups:
        encoded = encoder(prompt_embeddings, tokenized_prompts).mean(dim=0)
        features.append(encoded / encoded.norm())
    return torch.stack(features, dim=1)
