import torch
from torch import nn


def spatial_token_count(latent_size: int, group_width: int, projection_width: int, condition_dim: int) -> int:
    """Number of conditioning tokens the mask projection produces.

    The mask latent is flattened in NCHW order into rows of ``group_width``, each row is projected
    to ``projection_width``, and the result is reshaped into ``condition_dim``-wide tokens.
    """
    if latent_size % group_width:
        raise ValueError(f"latent size {latent_size} is not divisible by mask_group_width {group_width}")
    projected = (latent_size // group_width) * projection_width
    if projected % condition_dim:
        raise ValueError(f"projected size {projected} is not divisible by condition_dim {condition_dim}")
    return projected // condition_dim


class MaskProjection(nn.Module):
    """Maps a four-channel mask latent to spatial conditioning tokens.

    Mirrors ReplayCAD's ``mask_linear``: reshape to ``(B, -1, group_width)``, apply
    ``Linear + ReLU``, reshape to ``(B, -1, condition_dim)``, then concatenate with the text
    conditioning before cross-attention.
    """

    def __init__(self, group_width: int, projection_width: int, condition_dim: int):
        super().__init__()
        self.group_width = group_width
        self.condition_dim = condition_dim
        self.layers = nn.Sequential(nn.Linear(group_width, projection_width), nn.ReLU())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size = latents.shape[0]
        grouped = latents.reshape(batch_size, -1, self.group_width)
        projected = self.layers(grouped)
        return projected.reshape(batch_size, -1, self.condition_dim)
