import numpy as np
import pytest

from pyclad.vision.strategies.replaycad.artifacts import load_artifact
from pyclad.vision.strategies.replaycad.memory import ReplayCADMemory


def _images(count: int = 4) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(count, 8, 8, 3), dtype=np.uint8)


class _ExplodingMaskProvider:
    """A mask provider that fails loudly if it is ever asked for masks."""

    def masks_for(self, concept_id, images):
        raise AssertionError(f"masks_for('{concept_id}') should not be called when spatial conditioning is off")


def test_compress_writes_an_artifact(replaycad_config, stub_backend):
    config = replaycad_config()
    memory = ReplayCADMemory(config=config, backend=stub_backend(config), benchmark="mvtec")

    memory.compress("mvtec__screw", _images())

    artifact = load_artifact("mvtec__screw", config.artifact_dir / "mvtec")
    assert artifact is not None
    assert artifact.embedding.shape == (config.semantic_tokens, config.condition_dim)
    assert len(artifact.masks) == config.masks_per_concept
    assert memory.known_concepts() == ["mvtec__screw"]


def test_second_run_reuses_the_cached_artifact(replaycad_config, stub_backend):
    config = replaycad_config()
    backend = stub_backend(config)
    ReplayCADMemory(config=config, backend=backend, benchmark="mvtec").compress("mvtec__screw", _images())

    fresh_backend = stub_backend(config)
    ReplayCADMemory(config=config, backend=fresh_backend, benchmark="mvtec").compress("mvtec__screw", _images())

    assert fresh_backend.compress_calls == []  # cache hit: stage 1 not re-run


def test_changed_compression_config_recompresses(replaycad_config, stub_backend, caplog):
    first = replaycad_config()
    ReplayCADMemory(config=first, backend=stub_backend(first), benchmark="mvtec").compress("mvtec__screw", _images())

    second = replaycad_config(compression_steps=3)
    backend = stub_backend(second)
    with caplog.at_level("WARNING"):
        ReplayCADMemory(config=second, backend=backend, benchmark="mvtec").compress("mvtec__screw", _images())

    assert backend.compress_calls == ["mvtec__screw"]
    assert "config_hash" in caplog.text


def test_strict_cache_raises_on_a_hash_mismatch(replaycad_config, stub_backend):
    first = replaycad_config()
    ReplayCADMemory(config=first, backend=stub_backend(first), benchmark="mvtec").compress("mvtec__screw", _images())

    second = replaycad_config(compression_steps=3, strict_cache=True)
    memory = ReplayCADMemory(config=second, backend=stub_backend(second), benchmark="mvtec")

    with pytest.raises(ValueError, match="strict_cache"):
        memory.compress("mvtec__screw", _images())


def test_generate_previous_covers_every_known_concept(replaycad_config, stub_backend):
    config = replaycad_config(replay_samples_per_concept=3)
    backend = stub_backend(config)
    memory = ReplayCADMemory(config=config, backend=backend, benchmark="mvtec")
    memory.compress("mvtec__screw", _images())
    memory.compress("mvtec__pill", _images())

    replay = memory.generate_previous()

    assert replay.shape == (6, config.resolution, config.resolution, 3)
    assert [call[0] for call in backend.generate_calls] == ["mvtec__screw", "mvtec__pill"]
    assert all(call[1] == 3 for call in backend.generate_calls)


def test_artifacts_are_stored_under_their_benchmark(replaycad_config, stub_backend):
    # Two benchmarks sharing one artifact_dir must not collide on a common category name.
    config = replaycad_config()
    ReplayCADMemory(config=config, backend=stub_backend(config), benchmark="mvtec").compress("screw", _images())
    ReplayCADMemory(config=config, backend=stub_backend(config), benchmark="btech").compress("screw", _images())

    assert (config.artifact_dir / "mvtec" / "screw" / "meta.json").is_file()
    assert (config.artifact_dir / "btech" / "screw" / "meta.json").is_file()


def test_a_benchmark_gets_its_own_cache(replaycad_config, stub_backend):
    config = replaycad_config()
    ReplayCADMemory(config=config, backend=stub_backend(config), benchmark="mvtec").compress("screw", _images())

    other = stub_backend(config)
    ReplayCADMemory(config=config, backend=other, benchmark="btech").compress("screw", _images())

    # Same concept name, same config, different benchmark: this must be a miss, not a hit.
    assert other.compress_calls == ["screw"]


def test_generate_previous_is_empty_before_any_compression(replaycad_config, stub_backend):
    config = replaycad_config()
    memory = ReplayCADMemory(config=config, backend=stub_backend(config), benchmark="mvtec")

    assert len(memory.generate_previous()) == 0


def test_generation_seed_differs_per_concept_but_is_reproducible(replaycad_config, stub_backend):
    config = replaycad_config()
    backend = stub_backend(config)
    memory = ReplayCADMemory(config=config, backend=backend, benchmark="mvtec")
    memory.compress("mvtec__screw", _images())
    memory.compress("mvtec__pill", _images())

    first = memory.generate_previous()
    seeds = [call[2] for call in backend.generate_calls]
    second = memory.generate_previous()

    assert seeds[0] != seeds[1]  # otherwise both concepts replay identical noise
    assert np.array_equal(first, second)


def test_masks_are_not_computed_when_spatial_conditioning_is_off(replaycad_config, stub_backend):
    """For semantic-only classes with mask_backend='sam', computing masks at all would be a full
    vit_h pass over every training image whose output the backend then ignores. The mask provider
    must not even be called, and compression must still succeed with the zero-length array that
    replaces it.
    """
    config = replaycad_config(use_spatial_conditioning=False)
    memory = ReplayCADMemory(
        config=config, backend=stub_backend(config), benchmark="mvtec", mask_provider=_ExplodingMaskProvider()
    )

    memory.compress("mvtec__capsules", _images())  # must not raise

    artifact = load_artifact("mvtec__capsules", config.artifact_dir / "mvtec")
    assert artifact is not None
    assert artifact.masks.shape[0] == 0
    assert memory.known_concepts() == ["mvtec__capsules"]


def test_release_is_forwarded_to_the_backend(replaycad_config, stub_backend):
    config = replaycad_config()
    backend = stub_backend(config)

    ReplayCADMemory(config=config, backend=backend, benchmark="mvtec").release_device_memory()

    assert backend.release_calls == 1
