from pathlib import Path

import pytest
from pydantic import ValidationError

from pyclad.vision.strategies.replaycad.config import ReplayCADConfig
from pyclad.vision.strategies.replaycad.per_class import (
    AUTHORS_OVERRIDES,
    apply_per_class,
)


def _base(benchmark: str) -> ReplayCADConfig:
    return ReplayCADConfig.for_benchmark(benchmark, artifact_dir=Path("/tmp/a"), mask_backend="full-frame")


def test_every_released_class_is_covered():
    assert len(AUTHORS_OVERRIDES["mvtec"]) == 14
    assert len(AUTHORS_OVERRIDES["visa"]) == 11


def test_mvtec_texture_classes_use_low_guidance():
    config = apply_per_class(_base("mvtec"), "mvtec__carpet", benchmark="mvtec")

    assert config.guidance_scale == 1.0
    assert config.compression_steps == 9999
    assert config.replay_samples_per_concept == 200
    assert config.initializer_word == "screw"


def test_grid_uses_the_three_direction_augmentation_and_scale_five():
    config = apply_per_class(_base("mvtec"), "grid", benchmark="mvtec")

    assert config.guidance_scale == 5.0
    assert config.replay_samples_per_concept == 800
    assert config.train_augmentation == "rotate_3_directions"


def test_applying_an_override_still_validates_the_shapes(monkeypatch):
    """The published table is self-consistent, so this injects an override that genuinely breaks."""
    from pyclad.vision.strategies.replaycad import per_class as module

    original_profile_fields = module._profile_fields

    def broken_profile(profile, wide_projection):
        fields = original_profile_fields(profile, wide_projection)
        fields["mask_projection_width"] = 196  # 196 does not divide into 768-wide tokens
        return fields

    monkeypatch.setattr(module, "_profile_fields", broken_profile)

    # model_copy() would let this through silently; model_validate() must not.
    with pytest.raises(ValidationError, match="conditioning tokens"):
        apply_per_class(_base("visa"), "visa__pcb1", benchmark="visa")


def test_the_two_unmasked_visa_classes_keep_their_own_training_recipe():
    # capsules and macaroni2 used the stock textual-inversion configs: 40 vectors, base lr 5e-3,
    # and a single-group optimizer with no multiplier on the embedding.
    for concept in ("visa__capsules", "visa__macaroni2"):
        config = apply_per_class(_base("visa"), concept, benchmark="visa")

        assert config.semantic_tokens == 40
        assert config.base_learning_rate == pytest.approx(5e-3)
        assert config.semantic_lr_multiplier == pytest.approx(1.0)
        assert config.semantic_learning_rate == pytest.approx(config.spatial_learning_rate)

    # Everything else keeps the mask-conditioned recipe.
    pcb1 = apply_per_class(_base("visa"), "visa__pcb1", benchmark="visa")
    assert (pcb1.semantic_tokens, pcb1.semantic_lr_multiplier) == (20, 100.0)


def test_capsules_training_augmentation_matches_its_config():
    # v1-finetune.yaml sets rotate_180: true; macaroni2's config has no such key.
    assert AUTHORS_OVERRIDES["visa"]["capsules"].train_augmentation == "rotate_180"
    assert AUTHORS_OVERRIDES["visa"]["macaroni2"].train_augmentation == "none"


def test_visa_mixes_diffusion_families():
    sd = apply_per_class(_base("visa"), "visa__pcb1", benchmark="visa")
    ldm = apply_per_class(_base("visa"), "visa__candle", benchmark="visa")

    assert sd.profile == "sd-512" and sd.resolution == 512 and sd.condition_dim == 768
    assert ldm.profile == "ldm-256" and ldm.resolution == 256 and ldm.condition_dim == 1280


def test_wide_projection_classes():
    pcb4 = apply_per_class(_base("visa"), "visa__pcb4", benchmark="visa")

    assert (pcb4.mask_group_width, pcb4.mask_projection_width) == (512, 192)
    assert pcb4.spatial_tokens == 8


def test_cubic_timestep_sampling_only_for_the_lifang_configs():
    assert apply_per_class(_base("visa"), "pcb1", benchmark="visa").timestep_sampling == "cubic"
    assert apply_per_class(_base("visa"), "fryum", benchmark="visa").timestep_sampling == "cubic"
    # chewinggum omits average_t and cashew sets it to true, so both stay uniform.
    assert apply_per_class(_base("visa"), "chewinggum", benchmark="visa").timestep_sampling == "uniform"
    assert apply_per_class(_base("visa"), "cashew", benchmark="visa").timestep_sampling == "uniform"
    assert apply_per_class(_base("mvtec"), "screw", benchmark="mvtec").timestep_sampling == "uniform"


def test_semantic_only_classes_disable_spatial_conditioning():
    for concept in ("visa__macaroni2", "visa__capsules"):
        config = apply_per_class(_base("visa"), concept, benchmark="visa")
        assert config.use_spatial_conditioning is False


def test_classes_missing_from_the_scripts_fall_back_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        mvtec = apply_per_class(_base("mvtec"), "mvtec__zipper", benchmark="mvtec")
        visa = apply_per_class(_base("visa"), "visa__pipe_fryum", benchmark="visa")

    assert mvtec.compression_steps == 20_000  # dataset default
    assert visa.compression_steps == 30_000
    assert "zipper" in caplog.text and "pipe_fryum" in caplog.text


def test_unknown_benchmark_returns_the_config_unchanged():
    base = _base("mvtec")

    assert apply_per_class(base, "btech__01", benchmark="btech") is base


def test_multidataset_names_resolve_on_the_category():
    prefixed = apply_per_class(_base("mvtec"), "mvtec-ad__screw", benchmark="mvtec")
    bare = apply_per_class(_base("mvtec"), "screw", benchmark="mvtec")

    assert prefixed.compression_steps == bare.compression_steps == 19999


def test_mask_augmentation_matches_the_generation_dispatch_scripts():
    # txt2img_with_mask.py: class_name in ["metal_nut","screw","fryum","hazelnut"] -> random_rorate
    for concept in ("metal_nut", "screw", "hazelnut"):
        assert apply_per_class(_base("mvtec"), concept, benchmark="mvtec").mask_augmentation == "random_rotate"
    assert apply_per_class(_base("visa"), "fryum", benchmark="visa").mask_augmentation == "random_rotate"

    # class_name in ["grid"] -> random_3direcions_rotate
    assert apply_per_class(_base("mvtec"), "grid", benchmark="mvtec").mask_augmentation == "random_3directions_rotate"

    # transistor -> little_rorate_and_move() with every default
    transistor = apply_per_class(_base("mvtec"), "transistor", benchmark="mvtec")
    assert transistor.mask_augmentation == "little_rotate_and_move"
    assert (
        transistor.mask_transform_angles,
        transistor.mask_transform_distance,
        transistor.mask_transform_transpose,
    ) == (
        2,
        0.05,
        False,
    )

    # cashew -> little_rorate_and_move(angles=10)
    cashew = apply_per_class(_base("visa"), "cashew", benchmark="visa")
    assert (cashew.mask_augmentation, cashew.mask_transform_angles) == ("little_rotate_and_move", 10)

    # chewinggum -> little_rorate_and_move(distance=0.1, tranpose=True)
    chewinggum = apply_per_class(_base("visa"), "chewinggum", benchmark="visa")
    assert chewinggum.mask_augmentation == "little_rotate_and_move"
    assert (chewinggum.mask_transform_distance, chewinggum.mask_transform_transpose) == (0.1, True)

    # pcb1/pcb2/pcb3 -> little_rorate_and_move(distance=0.02, tranpose=True); pcb4 without tranpose
    for concept in ("pcb1", "pcb2", "pcb3"):
        pcb = apply_per_class(_base("visa"), concept, benchmark="visa")
        assert pcb.mask_augmentation == "little_rotate_and_move"
        assert (pcb.mask_transform_distance, pcb.mask_transform_transpose) == (0.02, True)
    pcb4 = apply_per_class(_base("visa"), "pcb4", benchmark="visa")
    assert pcb4.mask_augmentation == "little_rotate_and_move"
    assert (pcb4.mask_transform_distance, pcb4.mask_transform_transpose) == (0.02, False)

    # candle -> visa_candle(); macaroni1 -> visa_candle(rotate=True)
    candle = apply_per_class(_base("visa"), "candle", benchmark="visa")
    assert (candle.mask_augmentation, candle.visa_candle_shift_pixels, candle.visa_candle_rotate) == (
        "visa_candle",
        20,
        False,
    )
    macaroni1 = apply_per_class(_base("visa"), "macaroni1", benchmark="visa")
    assert (macaroni1.mask_augmentation, macaroni1.visa_candle_rotate) == ("visa_candle", True)

    # class_name in [...] -> pass (no transform at all)
    for concept in ("bottle", "cable", "capsule", "carpet", "leather", "pill", "toothbrush", "wood", "tile"):
        assert apply_per_class(_base("mvtec"), concept, benchmark="mvtec").mask_augmentation == "none"


def test_capsules_and_macaroni2_never_receive_a_mask_transform():
    # Both generate through the plain, maskless scripts (stable_txt2img.py / txt2img.py, per
    # generate_visa.sh), so neither ever reaches txt2img_with_mask.py / stable_txt2img_with_mask.py
    # and no generation-time mask transform is ever applied -- matching their
    # use_spatial_conditioning=False above. This holds regardless of the two release dispatchers'
    # own dead, mutually-contradictory macaroni2 branches (random_reset vs. random_rotate).
    for concept in ("visa__capsules", "visa__macaroni2"):
        config = apply_per_class(_base("visa"), concept, benchmark="visa")
        assert config.mask_augmentation == "none"
