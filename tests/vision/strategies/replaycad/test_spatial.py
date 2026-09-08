import pytest
import torch

from pyclad.vision.strategies.replaycad.spatial import (
    MaskProjection,
    spatial_token_count,
)


@pytest.mark.parametrize(
    "latent_size, group, projection, condition_dim, expected",
    [
        (32 * 32 * 4, 128, 200, 1280, 5),  # MVTec / LDM-256
        (64 * 64 * 4, 128, 192, 768, 32),  # VisA / SD-512
        (64 * 64 * 4, 512, 192, 768, 8),  # released pcb4 / fryum wide projection
    ],
)
def test_token_counts_match_the_reference_shapes(latent_size, group, projection, condition_dim, expected):
    assert spatial_token_count(latent_size, group, projection, condition_dim) == expected


def test_projection_maps_latents_to_condition_tokens():
    projection = MaskProjection(group_width=128, projection_width=200, condition_dim=1280)

    tokens = projection(torch.zeros(3, 4, 32, 32))

    assert tokens.shape == (3, 5, 1280)


def test_projection_is_a_linear_relu_stack():
    projection = MaskProjection(group_width=128, projection_width=200, condition_dim=1280)

    keys = list(projection.state_dict().keys())

    assert keys == ["layers.0.weight", "layers.0.bias"]


def test_projection_output_is_non_negative():
    projection = MaskProjection(group_width=128, projection_width=200, condition_dim=1280)

    tokens = projection(torch.randn(2, 4, 32, 32))

    assert torch.all(tokens >= 0)  # ReLU is the last op, as in ddpm2.py:528


def test_reshape_error_names_the_adjustable_fields():
    from pathlib import Path

    from pydantic import ValidationError

    from pyclad.vision.strategies.replaycad.config import ReplayCADConfig

    with pytest.raises(ValidationError, match="mask_projection_width"):
        ReplayCADConfig.for_benchmark(
            "visa", artifact_dir=Path("/tmp/a"), mask_backend="full-frame", mask_projection_width=196
        )
