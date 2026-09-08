from pathlib import Path

import numpy as np
import pytest
import torch

from pyclad.vision.strategies.replaycad.artifacts import (
    ConceptArtifact,
    compression_config_hash,
    concept_slug,
    load_artifact,
    save_artifact,
)
from pyclad.vision.strategies.replaycad.config import ReplayCADConfig


def _config(tmp_path: Path, **overrides) -> ReplayCADConfig:
    return ReplayCADConfig.for_benchmark("mvtec", artifact_dir=tmp_path, mask_backend="full-frame", **overrides)


def _artifact(config: ReplayCADConfig, concept_id: str = "mvtec__screw") -> ConceptArtifact:
    return ConceptArtifact(
        concept_id=concept_id,
        embedding=torch.arange(20 * 1280, dtype=torch.float32).reshape(20, 1280),
        projection_state={"layers.0.weight": torch.ones(200, 128), "layers.0.bias": torch.zeros(200)},
        masks=np.full((3, 8, 8), 255, dtype=np.uint8),
        config_hash=compression_config_hash(config),
        spatial_tokens=5,
    )


def test_artifact_round_trip(tmp_path: Path):
    config = _config(tmp_path)
    saved = save_artifact(_artifact(config), tmp_path)

    loaded = load_artifact("mvtec__screw", tmp_path)

    assert saved.is_dir()
    assert loaded is not None
    assert loaded.concept_id == "mvtec__screw"
    assert loaded.embedding.shape == (20, 1280)
    assert torch.equal(loaded.embedding, _artifact(config).embedding)
    assert loaded.masks.shape == (3, 8, 8)
    assert loaded.config_hash == compression_config_hash(config)
    assert loaded.spatial_tokens == 5


def test_projection_state_round_trips_with_its_module_keys(tmp_path: Path):
    # MaskProjection is nn.Sequential(nn.Linear, nn.ReLU), so these two keys are the contract that
    # lets a stored artifact be loaded straight back into the module.
    config = _config(tmp_path)
    original = _artifact(config)
    save_artifact(original, tmp_path)

    loaded = load_artifact("mvtec__screw", tmp_path)

    assert set(loaded.projection_state) == {"layers.0.weight", "layers.0.bias"}
    assert torch.equal(loaded.projection_state["layers.0.weight"], original.projection_state["layers.0.weight"])
    assert torch.equal(loaded.projection_state["layers.0.bias"], original.projection_state["layers.0.bias"])


def test_missing_artifact_returns_none(tmp_path: Path):
    assert load_artifact("mvtec__nothing", tmp_path) is None


def test_interrupted_resave_does_not_load_as_valid(tmp_path: Path, monkeypatch):
    # The dangerous case is re-saving a concept whose config did not change: the caller's hash check
    # would still match, so a half-written artifact has to fail to load rather than look like a hit.
    config = _config(tmp_path)
    save_artifact(_artifact(config), tmp_path)
    assert load_artifact("mvtec__screw", tmp_path) is not None

    import pyclad.vision.strategies.replaycad.artifacts as artifacts_module

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts_module.torch, "save", explode)

    with pytest.raises(OSError):
        save_artifact(_artifact(config), tmp_path)

    assert load_artifact("mvtec__screw", tmp_path) is None


def test_meta_write_is_atomic_so_a_crash_mid_write_fails_closed(tmp_path: Path, monkeypatch):
    """meta.json is itself the crash-safety marker load_artifact checks for. A crash while its own
    content is being written must not leave a truncated file behind: that would still pass the
    is_file() gate and then raise JSONDecodeError on every subsequent run, instead of the clean
    "no artifact" result a failed save is supposed to produce.
    """
    config = _config(tmp_path)

    import pyclad.vision.strategies.replaycad.artifacts as artifacts_module

    original_write_text = Path.write_text

    def crash_after_a_partial_write(self, content, *args, **kwargs):
        # Matches both the fixed code's temp file ("meta.json.tmp") and the pre-fix code's direct
        # write ("meta.json" itself), so this same test demonstrates the bug against either.
        if self.name in ("meta.json", "meta.json.tmp"):
            original_write_text(self, content[: len(content) // 2], *args, **kwargs)
            raise OSError("simulated crash mid-write")
        return original_write_text(self, content, *args, **kwargs)

    monkeypatch.setattr(artifacts_module.Path, "write_text", crash_after_a_partial_write)

    with pytest.raises(OSError):
        save_artifact(_artifact(config), tmp_path)

    # The crash must not leave anything at the real meta.json path -- a partially written file
    # there would pass load_artifact's existence check and then blow up with JSONDecodeError.
    assert not (tmp_path / "mvtec__screw" / "meta.json").exists()
    assert load_artifact("mvtec__screw", tmp_path) is None


def test_resave_with_fewer_masks_leaves_none_behind(tmp_path: Path):
    config = _config(tmp_path)
    save_artifact(_artifact(config), tmp_path)

    shrunk = _artifact(config)
    shrunk.masks = np.full((1, 8, 8), 255, dtype=np.uint8)
    save_artifact(shrunk, tmp_path)

    assert load_artifact("mvtec__screw", tmp_path).masks.shape[0] == 1


def test_multidataset_concept_names_are_slugged(tmp_path: Path):
    config = _config(tmp_path)
    save_artifact(_artifact(config, concept_id="visa/pcb 1"), tmp_path)

    assert concept_slug("visa/pcb 1") == "visa_pcb-1"
    assert load_artifact("visa/pcb 1", tmp_path) is not None


def test_hash_changes_with_compression_inputs(tmp_path: Path):
    base = compression_config_hash(_config(tmp_path))

    assert compression_config_hash(_config(tmp_path, compression_steps=1)) != base
    assert compression_config_hash(_config(tmp_path, semantic_tokens=8)) != base
    assert compression_config_hash(_config(tmp_path, mask_projection_width=400)) != base
    assert compression_config_hash(_config(tmp_path, seed=7)) != base
    assert compression_config_hash(_config(tmp_path, train_augmentation="rotate_180")) != base
    # weight_decay changes what compress() optimizes (a standalone AdamW-decayed parameter, not
    # the whole embedding matrix under a zeroed decay -- see backend.py's DiffusersBackend
    # docstring): an artifact compressed under one weight_decay must not cache-hit for another.
    assert compression_config_hash(_config(tmp_path, weight_decay=0.05)) != base


def test_hash_covers_where_the_masks_come_from(tmp_path: Path):
    # The mask source feeds the compression loop directly, so changing it must invalidate the cache.
    base = compression_config_hash(_config(tmp_path))

    assert compression_config_hash(_config(tmp_path, sam_checkpoint=tmp_path / "vit_h.pth")) != base
    assert compression_config_hash(_config(tmp_path, sam_model_type="vit_b")) != base
    assert compression_config_hash(_config(tmp_path, precomputed_mask_root=tmp_path / "sam")) != base


def test_hash_ignores_inputs_that_do_not_change_the_compressed_representation(tmp_path: Path):
    base = compression_config_hash(_config(tmp_path))

    # Ordering, detector and replay volume must not invalidate a compression pass.
    assert compression_config_hash(_config(tmp_path, replay_samples_per_concept=1)) == base
    assert compression_config_hash(_config(tmp_path, generation_batch_size=1)) == base
    assert compression_config_hash(_config(tmp_path, guidance_scale=5.0)) == base
    assert compression_config_hash(_config(tmp_path, strict_cache=True)) == base
    # Replay-time mask jitter cannot change an already-compressed artifact, so it must not
    # invalidate one. Hashing it would cost a full recompression per concept for nothing.
    assert compression_config_hash(_config(tmp_path, mask_augmentation="paper")) == base
    # Same for the five class-specific transforms and their parameters -- all replay-time.
    assert compression_config_hash(_config(tmp_path, mask_augmentation="little_rotate_and_move")) == base
    assert compression_config_hash(_config(tmp_path, mask_transform_angles=10)) == base
    assert compression_config_hash(_config(tmp_path, mask_transform_distance=0.2)) == base
    assert compression_config_hash(_config(tmp_path, mask_transform_transpose=True)) == base
    assert compression_config_hash(_config(tmp_path, visa_candle_shift_pixels=5)) == base
    assert compression_config_hash(_config(tmp_path, visa_candle_rotate=True)) == base


def test_artifact_without_spatial_conditioning_round_trips(tmp_path: Path):
    config = _config(tmp_path, use_spatial_conditioning=False)
    artifact = _artifact(config)
    artifact.projection_state = None
    artifact.masks = np.zeros((0, 8, 8), dtype=np.uint8)
    save_artifact(artifact, tmp_path)

    loaded = load_artifact("mvtec__screw", tmp_path)

    assert loaded is not None
    assert loaded.projection_state is None
    assert loaded.masks.shape[0] == 0
