from typing import Optional

import numpy as np
import pytest
import torch

from pyclad.vision.strategies.replaycad.artifacts import ConceptArtifact
from pyclad.vision.strategies.replaycad.config import ReplayCADConfig


class StubDiffusionBackend:
    """In-memory stand-in for the diffusion backend.

    Records every call so tests can assert on the strategy's call order, and returns
    deterministic tensors and images shaped exactly like the real backend's.
    """

    def __init__(self, config: ReplayCADConfig):
        self.config = config
        self.compress_calls: list[str] = []
        self.generate_calls: list[tuple[str, int, int]] = []
        self.release_calls = 0

    def compress(self, concept_id: str, images: np.ndarray, masks: np.ndarray):
        if len(images) == 0:
            raise ValueError(f"Cannot compress empty concept '{concept_id}'")
        # A zero-length mask array means spatial conditioning is off and no masks were computed,
        # matching the real backend's tolerance (see DiffusersBackend.compress).
        if len(masks) != 0 and len(images) != len(masks):
            raise ValueError(f"{len(images)} images but {len(masks)} masks for '{concept_id}'")
        self.compress_calls.append(concept_id)

        generator = torch.Generator().manual_seed(abs(hash(concept_id)) % (2**31))
        embedding = torch.randn(self.config.semantic_tokens, self.config.condition_dim, generator=generator)
        if not self.config.use_spatial_conditioning:
            return embedding, None
        projection_state = {
            "layers.0.weight": torch.zeros(self.config.mask_projection_width, self.config.mask_group_width),
            "layers.0.bias": torch.zeros(self.config.mask_projection_width),
        }
        return embedding, projection_state

    def generate(self, artifact: ConceptArtifact, count: int, seed: int) -> np.ndarray:
        self.generate_calls.append((artifact.concept_id, count, seed))
        rng = np.random.default_rng(seed)
        size = self.config.resolution
        return rng.integers(0, 256, size=(count, size, size, 3), dtype=np.uint8)

    def release_device_memory(self) -> None:
        self.release_calls += 1


@pytest.fixture
def replaycad_config(tmp_path):
    def build(**overrides) -> ReplayCADConfig:
        defaults = dict(
            mask_backend="full-frame",
            resolution=8,
            condition_dim=4,
            mask_group_width=2,
            mask_projection_width=2,
            semantic_tokens=3,
            compression_steps=2,
            compression_batch_size=1,
            replay_samples_per_concept=2,
            generation_batch_size=1,
            masks_per_concept=2,
        )
        defaults.update(overrides)
        return ReplayCADConfig.for_benchmark("mvtec", artifact_dir=tmp_path, **defaults)

    return build


@pytest.fixture
def stub_backend(replaycad_config):
    def build(config: Optional[ReplayCADConfig] = None) -> StubDiffusionBackend:
        return StubDiffusionBackend(config or replaycad_config())

    return build
