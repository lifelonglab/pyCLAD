import numpy as np
import pytest
import torch
from sklearn.neighbors import NearestNeighbors

from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.ucad import (
    UCAD,
    UCADTaskMemory,
    structure_contrastive_loss,
)
from pyclad.vision.models.utilities.base_model import VisionScoringBase
from pyclad.vision.prediction_results import VisionPredictionResults


def test_loss_matches_the_reference_formula_on_a_hand_computed_case():
    # Two orthogonal unit features in different regions. Normalised cosine similarities are
    # [[1, 0], [0, 1]]; divided by T=0.5 that is [[2, 0], [0, 2]]. The reference computes
    # (-sim * same_region + exp(sim) * (1 - same_region)).mean(), keeping the self-pairs:
    #   same-region diagonal: -2 and -2;  cross-region off-diagonal: exp(0) and exp(0)
    #   mean = (-2 + 1 + 1 - 2) / 4 = -0.5
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    labels = torch.tensor([[0, 1]])

    assert structure_contrastive_loss(features, labels, temperature=0.5).item() == pytest.approx(-0.5)


def test_identical_regions_give_a_purely_attractive_loss():
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    labels = torch.tensor([[0, 0]])

    # every pair is same-region: mean(-sim) = -(2 + 0 + 0 + 2) / 4 = -1.0
    assert structure_contrastive_loss(features, labels, temperature=0.5).item() == pytest.approx(-1.0)


def test_loss_is_invariant_to_feature_magnitude():
    features = torch.tensor([[[3.0, 0.0], [0.0, 7.0]]])
    labels = torch.tensor([[0, 1]])

    assert structure_contrastive_loss(features, labels, temperature=0.5).item() == pytest.approx(-0.5)


def test_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="region_labels must have shape"):
        structure_contrastive_loss(torch.zeros(1, 2, 2), torch.zeros(1, 3, dtype=torch.long))

    with pytest.raises(ValueError, match=r"patch_features must have shape"):
        structure_contrastive_loss(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))

    with pytest.raises(ValueError, match="temperature must be positive"):
        structure_contrastive_loss(torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.long), temperature=0.0)


timm = pytest.importorskip("timm")


def _config(**overrides) -> UCADConfig:
    """A ViT-Tiny at 32x32: a 2x2 patch grid, 4 patches per image. CPU-fast."""
    defaults = dict(
        backbone_name="vit_tiny_patch16_224",
        pretrained_backbone=False,
        input_size=(32, 32),
        input_range="float01",
        batch_size=2,
        feature_layer=3,
        prompt_depth=3,
        target_embed_dimension=16,
        key_size=3,
        knowledge_size=3,
        coreset_projection_dimension=4,
        coreset_starting_points=2,
        epochs=1,
        seed=0,
    )
    defaults.update(overrides)
    return UCADConfig(**defaults)


def _images(count: int, fill: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((count, 32, 32, 3), dtype=np.float32) * 0.1 + fill).astype(np.float32)


def test_backbone_is_frozen_and_the_prompt_is_the_only_trainable_tensor():
    model = UCAD(_config())

    assert all(not p.requires_grad for p in model.module.parameters())


def test_patch_features_have_the_mapped_dimension_and_a_square_grid():
    model = UCAD(_config())
    batch = model._preprocessor.transform(_images(2, 0.3)).to(model._device)

    features, grid = model._patch_features(batch, prompt=None)

    assert grid == (2, 2)
    assert features.shape == (2 * 4, 16)
    assert features.dtype == np.float32


def test_the_prompt_changes_the_features_and_receives_the_only_gradient():
    model = UCAD(_config())
    batch = model._preprocessor.transform(_images(2, 0.3)).to(model._device)

    unprompted = model._forward_patch_tokens(batch, None)
    prompt = torch.nn.Parameter(model._initial_prompt())
    prompted = model._forward_patch_tokens(batch, prompt)

    assert not torch.allclose(unprompted, prompted)

    prompted.sum().backward()
    assert prompt.grad is not None
    assert torch.isfinite(prompt.grad).all()
    assert all(p.grad is None for p in model.module.parameters())


def test_initial_prompt_has_the_prefix_key_value_shape():
    model = UCAD(_config(prompt_depth=3, prompt_length=1))
    heads = model.module.blocks[0].attn.num_heads

    assert tuple(model._initial_prompt().shape) == (3, 2, 1, heads, model.module.embed_dim // heads)


def test_zero_prompt_init_leaves_a_zero_tensor():
    model = UCAD(_config(prompt_init="zero"))

    assert torch.count_nonzero(model._initial_prompt()) == 0


def test_a_non_vit_backbone_is_rejected_with_a_clear_message():
    with pytest.raises(TypeError, match="compatible timm ViT"):
        UCAD(_config(backbone_name="resnet18"))


def test_feature_layer_beyond_the_backbone_depth_is_rejected():
    with pytest.raises(ValueError, match="exceeds backbone depth"):
        UCAD(_config(feature_layer=99, prompt_depth=1))


def test_fit_appends_one_task_memory_per_call_and_never_resets():
    model = UCAD(_config())

    model.set_current_concept("a")
    model.fit(_images(4, 0.2, seed=1))
    model.set_current_concept("b")
    model.fit(_images(4, 0.7, seed=2))

    assert len(model.task_memories) == 2
    assert [memory.concept_id for memory in model.task_memories] == ["a", "b"]


def test_key_and_knowledge_are_capped_at_the_configured_sizes():
    model = UCAD(_config(key_size=3, knowledge_size=2))
    model.fit(_images(4, 0.2))  # 4 images * 4 patches = 16 candidate vectors

    memory = model.task_memories[0]
    assert memory.key.shape == (3, 16)
    assert memory.knowledge.shape == (2, 16)


def test_all_patches_are_kept_when_there_are_fewer_than_the_target():
    model = UCAD(_config(key_size=999, knowledge_size=999))
    model.fit(_images(2, 0.2))  # 2 images * 4 patches = 8 vectors

    assert model.task_memories[0].key.shape == (8, 16)


def test_cpm_only_ablation_does_not_train_the_prompt():
    # prompt_init="zero" makes this deterministic without reasoning about RNG ordering: an
    # untrained zero prompt must still be all zeros after fit.
    model = UCAD(_config(structure_mode="none", epochs=5, prompt_init="zero"))

    model.fit(_images(4, 0.2))

    memory = model.task_memories[0]
    assert memory.final_scl_loss is None
    assert torch.count_nonzero(memory.prompt) == 0


def test_scl_training_moves_the_prompt_and_records_a_loss(tmp_path):
    from PIL import Image

    directory = tmp_path / "bottle"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(4):
        regions = rng.integers(0, 3, size=(32, 32)).astype(np.uint8)
        Image.fromarray(regions, mode="L").save(directory / f"{index:03d}.png")

    model = UCAD(
        _config(
            structure_mode="precomputed",
            structure_mask_root=str(tmp_path),
            epochs=2,
            prompt_init="zero",
        )
    )

    model.set_current_concept("bottle")
    model.fit(_images(4, 0.2))

    memory = model.task_memories[0]
    assert memory.final_scl_loss is not None
    assert np.isfinite(memory.final_scl_loss)
    assert torch.count_nonzero(memory.prompt) > 0  # a zero prompt that trained is no longer zero
    assert model.additional_info()["variant"] == "CPM+SCL"
    assert model.additional_info()["last_loss"] == memory.final_scl_loss


def test_train_prompt_rejects_structure_masks_with_the_wrong_spatial_shape():
    # A mask array shaped like NCHW-misread dimensions (e.g. (channels, height) instead of
    # (height, width)) must raise rather than silently train on garbage region labels.
    model = UCAD(_config())
    data = _images(2, 0.2)
    wrong_shape_masks = np.zeros((2, 3, 32), dtype=np.int64)

    with pytest.raises(ValueError, match=r"\(3, 32\).*\(32, 32\)"):
        model._train_prompt(data, wrong_shape_masks)


def test_fit_rejects_an_empty_task():
    model = UCAD(_config())

    with pytest.raises(ValueError, match="cannot learn an empty task"):
        model.fit(np.empty((0, 32, 32, 3), dtype=np.float32))


def test_tasks_must_share_a_patch_grid():
    # Mutating config.input_size would not work: the preprocessor captured its size at
    # construction. Set the recorded grid directly to stand in for an earlier task that had one.
    model = UCAD(_config())
    model.fit(_images(2, 0.2))
    model._grid_size = (7, 7)

    with pytest.raises(ValueError, match="must share the same feature grid"):
        model.fit(_images(2, 0.5))


def _two_task_model(**overrides) -> UCAD:
    model = UCAD(_config(**overrides))
    model.set_current_concept("dark")
    model.fit(_images(4, 0.05, seed=11))
    model.set_current_concept("bright")
    model.fit(_images(4, 0.85, seed=12))
    return model


def test_predict_returns_the_vision_contract():
    model = _two_task_model()

    result = model.predict(_images(3, 0.05, seed=21))

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (3,)
    assert result.anomaly_scores.shape == (3,)
    assert result.score_maps.shape == (3, 32, 32)


def test_routing_is_per_image_within_a_single_batch():
    model = _two_task_model(batch_size=8)
    mixed = np.concatenate([_images(3, 0.05, seed=31), _images(3, 0.85, seed=32)])

    selected = model.selected_task_indices(mixed)

    assert selected.shape == (6,)
    assert set(selected[:3]) == {0}
    assert set(selected[3:]) == {1}


def test_a_single_task_routes_everything_to_itself():
    model = UCAD(_config())
    model.fit(_images(4, 0.2))

    np.testing.assert_array_equal(model.selected_task_indices(_images(3, 0.9)), np.zeros(3, dtype=np.int64))


def test_reported_score_is_the_cached_raw_patch_maximum_and_the_cache_is_cleared_after_predict():
    # Smoothing with sigma=4 flattens a 2x2 map towards its mean, so the raw per-patch maximum
    # UCAD caches and reports must differ from what generic max-pooling over the (already
    # smoothed, already resized) returned maps would give -- that gap is real but only ~5e-4 on
    # this grid, far too thin a margin to rest a regression guard on. So pin the handshake
    # itself rather than a magnitude comparison: the reported score must not equal what
    # VisionScoringBase._aggregate_scores computes from the same maps, and the cache the
    # override consumes must not survive past predict() to leak into a later batch.
    model = UCAD(_config(smoothing_sigma=4.0))
    model.fit(_images(4, 0.2))

    result = model.predict(_images(2, 0.9, seed=41))

    generic_scores = VisionScoringBase._aggregate_scores(model, torch.from_numpy(result.score_maps))
    assert not np.array_equal(result.anomaly_scores, generic_scores.numpy())
    assert model._cached_image_scores is None


def test_patch_scores_are_squared_l2_like_the_reference_faiss_index():
    # The reference NearestNeighbourScorer is backed by faiss.IndexFlatL2, which returns SQUARED
    # distances; sklearn returns plain Euclidean. A knowledge bank at the origin and a query at
    # distance 3 must score 9, not 3. Two patches at clearly different distances (3 -> squared 9,
    # 1 -> squared 1) also pin that the image score is the MAX over patches (9), not their mean
    # (5) -- a 1x1 grid could not distinguish those, since max and mean coincide there.
    model = UCAD(_config(n_neighbors=1))
    model._grid_size = (1, 2)
    model._task_memories.append(
        UCADTaskMemory(
            key=np.zeros((1, 4), dtype=np.float32), prompt=torch.zeros(1), knowledge=np.zeros((1, 4), dtype=np.float32)
        )
    )
    model._key_indices.append(NearestNeighbors(n_neighbors=1).fit(np.zeros((1, 4), dtype=np.float32)))
    model._knowledge_indices.append(NearestNeighbors(n_neighbors=1).fit(np.zeros((1, 4), dtype=np.float32)))
    model._threshold = 0.0
    model._patch_features = lambda images, prompt: (
        np.array([[3.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        (1, 2),
    )

    result = model.predict(_images(1, 0.2))

    assert result.anomaly_scores[0] == pytest.approx(9.0)


def test_predict_before_fit_raises():
    model = UCAD(_config())

    with pytest.raises(RuntimeError, match="must learn at least one task"):
        model.predict(_images(2, 0.2))


def test_predict_handles_empty_input():
    model = UCAD(_config(threshold=0.0))
    model.fit(_images(2, 0.2))

    result = model.predict(np.empty((0, 32, 32, 3), dtype=np.float32))

    assert result.y_pred.shape == (0,)


def test_additional_info_reports_the_memory_and_the_variant():
    model = _two_task_model()

    info = model.additional_info()

    assert info["tasks_seen"] == 2
    assert info["task_concepts"] == ["dark", "bright"]
    assert info["variant"] == "CPM-only"
    assert info["key_prompt_knowledge_bytes"] > 0


def test_variant_is_cpm_only_when_epochs_is_zero_even_with_structure_mode_precomputed(tmp_path):
    # structure_mode="precomputed" is configuration intent; epochs=0 makes _train_prompt return
    # an untrained prompt regardless, so no SCL actually ran. The reported variant must say so.
    from PIL import Image

    directory = tmp_path / "bottle"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(4):
        regions = rng.integers(0, 3, size=(32, 32)).astype(np.uint8)
        Image.fromarray(regions, mode="L").save(directory / f"{index:03d}.png")

    model = UCAD(
        _config(
            structure_mode="precomputed",
            structure_mask_root=str(tmp_path),
            epochs=0,
        )
    )

    model.set_current_concept("bottle")
    model.fit(_images(4, 0.2))

    assert model.task_memories[0].final_scl_loss is None
    assert model.additional_info()["variant"] == "CPM-only"
