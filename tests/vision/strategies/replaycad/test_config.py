from pathlib import Path

import pytest
from pydantic import ValidationError

from pyclad.vision.strategies.replaycad.config import (
    IMAGENET_TEMPLATES_SMALL,
    ReplayCADConfig,
)


def _config(benchmark: str, **overrides) -> ReplayCADConfig:
    # mask_backend defaults to "sam", which requires a checkpoint. These tests are about presets
    # and derived arithmetic, so they pick the backend that needs no external file.
    defaults = {"mask_backend": "full-frame"}
    defaults.update(overrides)
    return ReplayCADConfig.for_benchmark(benchmark, artifact_dir=Path("/tmp/a"), **defaults)


def test_sam_backend_requires_a_checkpoint():
    # Failing loudly is deliberate: silently falling back to full-frame masks would destroy the
    # spatial conditioning without any signal.
    with pytest.raises(ValidationError, match="sam_checkpoint"):
        ReplayCADConfig.for_benchmark("mvtec", artifact_dir=Path("/tmp/a"))


def test_mvtec_preset_matches_the_paper():
    config = _config("mvtec")

    assert config.model_id == "CompVis/ldm-text2im-large-256"
    assert config.resolution == 256
    assert config.condition_dim == 1280
    assert config.semantic_tokens == 20
    assert (config.mask_group_width, config.mask_projection_width) == (128, 200)
    assert config.compression_steps == 20_000
    assert config.spatial_tokens == 5


def test_visa_preset_matches_the_released_sd_config():
    config = _config("visa")

    assert config.resolution == 512
    assert config.condition_dim == 768
    # The paper prints (128, 196), which cannot be reshaped into 768-dim tokens.
    assert (config.mask_group_width, config.mask_projection_width) == (128, 192)
    assert config.compression_steps == 30_000
    assert config.spatial_tokens == 32


@pytest.mark.parametrize(
    "benchmark, spatial_lr, semantic_lr",
    [("mvtec", 1.6e-3, 1.6e-1), ("visa", 2.0e-4, 2.0e-2)],
)
def test_learning_rates_reproduce_the_scaled_originals(benchmark, spatial_lr, semantic_lr):
    # main.py:770 -> lr = accumulate * ngpu * batch_size * base_lr, --scale_lr defaults to True,
    # every published run used --gpus 0,1. MVTec: 1*2*16*5e-5. VisA: 1*2*2*5e-5.
    config = _config(benchmark)

    assert config.spatial_learning_rate == pytest.approx(spatial_lr)
    assert config.semantic_learning_rate == pytest.approx(semantic_lr)


def test_unknown_benchmark_falls_back_to_the_ldm_profile():
    config = _config("btech")

    assert config.profile == "ldm-256"
    assert config.initializer_word is None


def test_non_reshapeable_projection_is_rejected():
    with pytest.raises(ValidationError, match="conditioning tokens"):
        _config("visa", mask_projection_width=196)


def test_indivisible_group_width_is_rejected():
    with pytest.raises(ValidationError, match="mask_group_width"):
        _config("mvtec", mask_group_width=97)


def test_masks_per_concept_is_capped_at_ten():
    # Matching on the field name keeps this from passing on some unrelated validation error.
    with pytest.raises(ValidationError, match="masks_per_concept"):
        _config("mvtec", masks_per_concept=11)


def test_defaults_follow_the_paper_not_the_released_scripts():
    config = _config("mvtec")

    assert config.replay_samples_per_concept == 800  # paper section 5.1
    assert config.guidance_scale == 10.0
    assert config.inference_steps == 50
    assert config.ddim_eta == 0.0
    assert config.mask_augmentation == "none"  # opt in to "paper" or one of the five class transforms
    assert config.initializer_word is None  # paper: randomly initialized embedding
    assert config.timestep_sampling == "uniform"  # ddpm2.py average_t defaults to True
    assert config.train_augmentation == "none"
    # mask_transfor.py's own defaults for the five class-specific transforms.
    assert config.mask_transform_angles == 2
    assert config.mask_transform_distance == 0.05
    assert config.mask_transform_transpose is False
    assert config.visa_candle_shift_pixels == 20
    assert config.visa_candle_rotate is False


@pytest.mark.parametrize(
    "mode",
    [
        "none",
        "paper",
        "random_rotate",
        "random_3directions_rotate",
        "little_rotate_and_move",
        "visa_candle",
        "random_reset",
    ],
)
def test_mask_augmentation_accepts_the_paper_mode_and_the_five_class_transforms(mode):
    assert _config("mvtec", mask_augmentation=mode).mask_augmentation == mode


def test_mask_augmentation_rejects_unknown_modes():
    with pytest.raises(ValidationError, match="mask_augmentation"):
        _config("mvtec", mask_augmentation="rotate_and_shift")


def test_prompt_templates_are_the_textual_inversion_list():
    assert len(IMAGENET_TEMPLATES_SMALL) == 81
    assert "a photo of a {}" in IMAGENET_TEMPLATES_SMALL
