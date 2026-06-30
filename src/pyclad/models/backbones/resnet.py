import torch
from pydantic import BaseModel, Field
from torch import nn

from pyclad.models.feature_extractor import FeatureExtractor


class ResNetBlockConfig(BaseModel):
    """Configuration for a single residual block for _ResNetBlock class.

    :param out_channels: Channel count produced by every conv layer in this block.
    :param kernel_sizes: Temporal kernel widths applied sequentially within the
        block. Defaults to ``[8, 5, 3]``, matching the reference CARLA repo.
    """

    out_channels: int
    kernel_sizes: list[int] = Field(default_factory=lambda: [8, 5, 3])


class _ResNetBlock(nn.Module):
    """Residual block built from a :class:`ResNetBlockConfig`."""

    def __init__(self, in_channels: int, config: ResNetBlockConfig) -> None:
        super().__init__()

        channels = [in_channels, *([config.out_channels] * len(config.kernel_sizes))]
        layers: list[nn.Module] = []
        for i, k in enumerate(config.kernel_sizes):
            layers.extend(
                (
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=k, padding="same"),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                )
            )
        self.layers = nn.Sequential(*layers)

        self.skip: nn.Module = (
            nn.Sequential(
                nn.Conv1d(in_channels, config.out_channels, kernel_size=1),
                nn.BatchNorm1d(config.out_channels),
            )
            if in_channels != config.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip(x)


def _default_blocks() -> list[ResNetBlockConfig]:
    return [
        ResNetBlockConfig(out_channels=4),
        ResNetBlockConfig(out_channels=8),
        ResNetBlockConfig(out_channels=8),
    ]


class ResNet1D(nn.Module, FeatureExtractor):
    """1D ResNet backbone for multivariate time series.

    :param num_features: Number of input channels (features per timestep).
    :param blocks: Per-block configuration.
    """

    def __init__(
        self,
        num_features: int,
        blocks: list[ResNetBlockConfig] | None = None,
    ) -> None:
        super().__init__()

        if blocks is None:
            blocks = _default_blocks()
        if not blocks:
            err_msg = "ResNet1D requires at least one block."
            raise ValueError(err_msg)

        in_channels_per_block = [num_features, *(b.out_channels for b in blocks[:-1])]
        self.blocks = nn.Sequential(*(_ResNetBlock(in_ch, cfg) for in_ch, cfg in zip(in_channels_per_block, blocks)))
        self.output_dim = blocks[-1].out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        return x.mean(dim=2)
