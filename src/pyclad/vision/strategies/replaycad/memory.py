from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pyclad.vision.strategies.replaycad.artifacts import (
    ConceptArtifact,
    compression_config_hash,
    load_artifact,
    save_artifact,
)
from pyclad.vision.strategies.replaycad.backend import DiffusionBackend
from pyclad.vision.strategies.replaycad.config import ReplayCADConfig
from pyclad.vision.strategies.replaycad.masks import (
    MaskProvider,
    build_mask_provider,
    select_stored_masks,
)

logger = logging.getLogger(__name__)


class ReplayCADMemory:
    """Holds ReplayCAD's compressed conditional features and replays historical concepts.

    Artifacts are cached under ``config.artifact_dir``. The cache key covers only inputs that
    change the compressed representation, so one compression pass is reused across concept
    orderings, detectors and evaluation seeds.
    """

    def __init__(
        self,
        config: ReplayCADConfig,
        backend: DiffusionBackend,
        benchmark: str = "",
        mask_provider: Optional[MaskProvider] = None,
    ):
        self.config = config
        self.backend = backend
        self.benchmark = benchmark
        self.mask_provider = mask_provider or build_mask_provider(config, benchmark)
        self._concepts: List[str] = []

    @property
    def _root(self) -> Path:
        """Artifact root for this benchmark.

        Concept ids carry no dataset prefix in a single-benchmark run, so without this, two
        benchmarks sharing one artifact_dir could collide on a common category name -- undetected,
        since config_hash doesn't cover benchmark identity.
        """
        return Path(self.config.artifact_dir) / self.benchmark if self.benchmark else Path(self.config.artifact_dir)

    def known_concepts(self) -> List[str]:
        return list(self._concepts)

    def release_device_memory(self) -> None:
        self.backend.release_device_memory()

    def compress(self, concept_id: str, images: np.ndarray) -> None:
        expected_hash = compression_config_hash(self.config)
        cached = load_artifact(concept_id, self._root)

        if cached is not None and cached.config_hash == expected_hash:
            logger.info("ReplayCAD reusing cached artifact for '%s'", concept_id)
            self._remember(concept_id)
            return

        if cached is not None:
            message = (
                f"Cached ReplayCAD artifact for '{concept_id}' was compressed with a different "
                f"config_hash ({cached.config_hash} != {expected_hash})."
            )
            if self.config.strict_cache:
                raise ValueError(f"{message} strict_cache=True refuses to recompress.")
            logger.warning("%s Recompressing.", message)

        if self.config.use_spatial_conditioning:
            masks = self.mask_provider.masks_for(concept_id, images)
        else:
            # Skipped when off: the backend ignores masks entirely, and for mask_backend="sam"
            # computing them anyway would mean a full vit_h pass over every image, discarded.
            masks = np.zeros((0, images.shape[1], images.shape[2]), dtype=np.uint8)

        embedding, projection_state = self.backend.compress(concept_id, images, masks)

        rng = np.random.default_rng(self.config.seed)
        stored_masks = (
            select_stored_masks(masks, self.config.masks_per_concept, rng)
            if self.config.use_spatial_conditioning
            else masks
        )

        save_artifact(
            ConceptArtifact(
                concept_id=concept_id,
                embedding=embedding,
                projection_state=projection_state,
                masks=stored_masks,
                config_hash=expected_hash,
                spatial_tokens=self.config.spatial_tokens if self.config.use_spatial_conditioning else 0,
            ),
            self._root,
        )
        self._remember(concept_id)

    def generate_previous(self) -> np.ndarray:
        if not self._concepts:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)

        batches = []
        for index, concept_id in enumerate(self._concepts):
            artifact = load_artifact(concept_id, self._root)
            if artifact is None:
                raise RuntimeError(f"ReplayCAD artifact for '{concept_id}' disappeared from the cache")
            # Distinct per-concept seed: a shared seed would replay identical latent noise for
            # every concept. Deterministic across runs because it is derived, not drawn.
            batches.append(
                self.backend.generate(
                    artifact, count=self.config.replay_samples_per_concept, seed=self.config.seed + index
                )
            )
        return np.concatenate(batches, axis=0)

    def _remember(self, concept_id: str) -> None:
        if concept_id not in self._concepts:
            self._concepts.append(concept_id)

    def info(self) -> Dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "profile": self.config.profile,
            "semantic_tokens": self.config.semantic_tokens,
            "spatial_tokens": self.config.spatial_tokens if self.config.use_spatial_conditioning else 0,
            "compression_steps": self.config.compression_steps,
            "spatial_learning_rate": self.config.spatial_learning_rate,
            "semantic_learning_rate": self.config.semantic_learning_rate,
            "guidance_scale": self.config.guidance_scale,
            "inference_steps": self.config.inference_steps,
            "replay_samples_per_concept": self.config.replay_samples_per_concept,
            "mask_backend": self.config.mask_backend,
            "mask_augmentation": self.config.mask_augmentation,
            "masks_per_concept": self.config.masks_per_concept,
            "artifact_dir": str(self.config.artifact_dir),
            "compressed_concepts": self.known_concepts(),
        }
