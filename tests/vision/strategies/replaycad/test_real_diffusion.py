"""End-to-end smoke test for ReplayCADStrategy against a real ``diffusers`` pipeline.

Every other test in this suite drives :class:`DiffusersBackend` through the in-memory
``StubDiffusionBackend`` (see ``conftest.py``). That stub returns tensors and images shaped
exactly like the real backend's, but it never calls into ``diffusers``/``transformers`` -- it
cannot catch a wrong component attribute name, an unsupported ``resize_token_embeddings`` call, a
renamed scheduler keyword argument, or a VAE that reshapes differently than assumed. This test
runs the real backend against a genuine, few-MB Stable-Diffusion-shaped pipeline instead, so those
assumptions get checked at least once.

It downloads that pipeline from the Hugging Face Hub on first run (cached afterwards), so it is
marked ``longrun`` and stays out of the default suite -- run it with ``pytest --longrun``.

A green result verifies the API contract, not the quality of the method: the pipeline is
Stable-Diffusion-shaped, so the LDM-256 path is exercised only by the unit-level stubs in
``test_backend_contract.py``, and no number from the paper is reproduced here.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyclad.vision.models.patchcore.config import PatchCoreConfig
from pyclad.vision.models.patchcore.patchcore import PatchCore
from pyclad.vision.strategies.replaycad.artifacts import load_artifact
from pyclad.vision.strategies.replaycad.backend import DiffusersBackend
from pyclad.vision.strategies.replaycad.config import ReplayCADConfig
from pyclad.vision.strategies.replaycad.memory import ReplayCADMemory
from pyclad.vision.strategies.replaycad.strategy import ReplayCADStrategy

# hf-internal-testing/tiny-stable-diffusion-pipe -- the brief's suggested first choice -- publishes
# only Flax weights (FlaxCLIPTextModel, FlaxUNet2DConditionModel, FlaxAutoencoderKL) under a
# `_class_name` of "StableDiffusionPipeline". Loading it through installed diffusers 0.40 fails
# before any pyCLAD code runs: `AttributeError: module diffusers.pipelines.stable_diffusion has no
# attribute FlaxStableDiffusionSafetyChecker`. diffusers/tiny-stable-diffusion-torch is diffusers'
# own torch-native counterpart in the same "tiny-*" test-fixture family (published under the
# `diffusers` org itself, for this exact purpose) and loads cleanly, so it is used instead.
MODEL_ID = "diffusers/tiny-stable-diffusion-torch"

# True dimensions read from the loaded pipeline (`pipe = DiffusionPipeline.from_pretrained(MODEL_ID,
# safety_checker=None)`), not assumed -- see task-12-report.md Step 2 for the full transcript:
#   pipe.unet.config.cross_attention_dim -> 32
#   pipe.unet.config.sample_size         -> 64 (StableDiffusionPipeline.__init__ floors any
#                                          checkpoint's sample_size to 64; the raw unet/config.json
#                                          on the Hub says 32, but 64 is what the loaded pipeline
#                                          object -- what backend.py actually sees -- reports)
#   pipe.unet.config.in_channels         -> 4
#   sorted(pipe.components)              -> feature_extractor, image_encoder, safety_checker,
#                                            scheduler, text_encoder, tokenizer, unet, vae
CONDITION_DIM = 32
RESOLUTION = 64

# mask_group_width=64, mask_projection_width=32 divide the *configured* latent arithmetic exactly,
# matching the brief's own worked example: latent_size = (64 // 8) ** 2 * 4 = 256; 256 / 64 = 4
# rows; 4 * 32 = 128; 128 / 32 = 4 spatial tokens. Nothing at runtime depends on that token count
# being accurate (MaskProjection reshapes whatever it is actually given), which is fortunate: this
# tiny VAE only has two down-sampling blocks and empirically halves resolution instead of the
# standard LDM/SD eighth, so the *real* per-mask latent has 4x the elements this arithmetic assumes.
# See "What remains unverified" in task-12-report.md.
MASK_GROUP_WIDTH = 64
MASK_PROJECTION_WIDTH = 32


class RecordingPatchCore(PatchCore):
    """Records how many images each ``fit`` call receives, without instrumenting production code."""

    def __init__(self, config: PatchCoreConfig):
        super().__init__(config)
        self.fit_sizes: list[int] = []

    def fit(self, data: np.ndarray) -> None:
        self.fit_sizes.append(len(data))
        super().fit(data)


def _images(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(count, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)


def _replaycad_config(tmp_path) -> ReplayCADConfig:
    return ReplayCADConfig(
        profile="ldm-256",
        model_id=MODEL_ID,
        resolution=RESOLUTION,
        condition_dim=CONDITION_DIM,
        semantic_tokens=2,
        mask_group_width=MASK_GROUP_WIDTH,
        mask_projection_width=MASK_PROJECTION_WIDTH,
        compression_steps=2,
        compression_batch_size=2,
        learning_rate_scale=1,
        inference_steps=2,
        replay_samples_per_concept=2,
        generation_batch_size=2,
        masks_per_concept=1,
        mask_backend="full-frame",
        artifact_dir=tmp_path,
        device="cpu",
        torch_dtype="float32",
    )


def _detector() -> RecordingPatchCore:
    # Small and untrained, matching tests/vision/models/test_patchcore.py's own pattern: this test
    # is about ReplayCAD's diffusion plumbing, not PatchCore's detection quality, and an untrained
    # resnet18 needs no download of its own.
    config = PatchCoreConfig(
        input_size=(32, 32),
        batch_size=4,
        backbone_name="resnet18",
        pretrained_backbone=False,
        coreset_sampling_ratio=1.0,
        pretrain_embed_dimension=32,
        target_embed_dimension=32,
        device="cpu",
        seed=0,
    )
    return RecordingPatchCore(config)


@pytest.mark.longrun
def test_replaycad_trains_generates_and_detects_on_a_real_diffusion_pipeline(tmp_path):
    config = _replaycad_config(tmp_path)
    memory = ReplayCADMemory(config=config, backend=DiffusersBackend(config))
    detector = _detector()
    strategy = ReplayCADStrategy(detector, memory)

    first_images = _images(2, seed=1)
    strategy.learn(first_images, concept_id="screw")

    second_images = _images(2, seed=2)
    strategy.learn(second_images, concept_id="pill")

    # The first concept has nothing to replay; the second trains on screw's replay plus pill's own
    # images -- the whole reason ReplayCAD exists.
    assert detector.fit_sizes[0] == 2
    assert detector.fit_sizes[1] == config.replay_samples_per_concept + 2

    for concept_id in ("screw", "pill"):
        concept_dir = tmp_path / concept_id
        assert (concept_dir / "embedding.pt").is_file()
        assert (concept_dir / "projection.pt").is_file()

    artifact = load_artifact("pill", tmp_path)
    assert artifact is not None
    assert artifact.embedding.shape == (config.semantic_tokens, config.condition_dim)

    result = strategy.predict(second_images, concept_id="pill")
    assert result.anomaly_scores.shape == (2,)
    assert np.all(np.isfinite(result.anomaly_scores))
    assert result.score_maps.shape == (2, RESOLUTION, RESOLUTION)
    assert np.all(np.isfinite(result.score_maps))
