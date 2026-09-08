import pytest

from pyclad.vision.models.ucad.config import UCADConfig


def test_defaults_match_the_reference_implementation():
    config = UCADConfig()

    assert config.backbone_name == "vit_base_patch16_224.augreg_in21k_ft_in1k"
    assert config.input_size == (224, 224)
    assert config.batch_size == 8
    assert config.feature_layer == 6
    assert config.prompt_depth == 6
    assert config.prompt_length == 1
    assert config.prompt_init == "uniform"
    assert config.target_embed_dimension == 1024
    assert config.key_size == 196
    assert config.knowledge_size == 196
    assert config.coreset_projection_dimension == 128
    assert config.coreset_starting_points == 10
    assert config.n_neighbors == 1
    assert config.epochs == 25
    assert config.learning_rate == pytest.approx(5e-4)
    assert config.weight_decay == 0.0
    assert config.gradient_clip_norm == pytest.approx(1.0)
    assert config.contrastive_temperature == pytest.approx(0.5)
    assert config.smoothing_sigma == pytest.approx(4.0)
    assert config.score_mode == "max"
    assert config.structure_mode == "none"
    assert config.mask_interpolation == "bilinear"
    assert config.routing_statistic == "mean"
    assert config.normalize_mean == (0.485, 0.456, 0.406)


def test_prompt_depth_may_exceed_feature_layer_like_the_reference():
    # The reference prompts all 12 blocks while reading features after block index 5. Blocks
    # past the read-out point receive no gradient, so this is wasteful but valid -- and it is
    # the configuration the authors actually ran. Rejecting it would make fidelity unreachable.
    config = UCADConfig(prompt_depth=12, feature_layer=6)

    assert config.prompt_depth == 12


def test_precomputed_mode_requires_an_existing_mask_root(tmp_path):
    with pytest.raises(ValueError, match="structure_mask_root"):
        UCADConfig(structure_mode="precomputed")

    with pytest.raises(ValueError, match="does not exist"):
        UCADConfig(structure_mode="precomputed", structure_mask_root=str(tmp_path / "nope"))

    UCADConfig(structure_mode="precomputed", structure_mask_root=str(tmp_path))


def test_sam_mode_requires_an_existing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="sam_checkpoint"):
        UCADConfig(structure_mode="sam")

    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"stub")
    UCADConfig(structure_mode="sam", sam_checkpoint=str(checkpoint))
