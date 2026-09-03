from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from pyclad.models.model import Model
from pyclad.strategies.strategy import ConceptAwareStrategy
from pyclad.vision.prediction_results import VisionPredictionResults
from pyclad.vision.strategies.replaycad.memory import ReplayCADMemory


class ReplayCADStrategy(ConceptAwareStrategy):
    """Diffusion generative replay for continual anomaly detection (Hu et al., IJCAI 2025).

    Per concept: replay every historical concept from its stored conditional features, refit the
    detector on replay + new data, then compress the new concept for future replay. The detector is
    an ordinary pyCLAD vision model -- the paper imposes no restriction on it.
    """

    def __init__(self, model: Model, memory: ReplayCADMemory):
        self._model = model
        self._memory = memory

    def learn(self, data: np.ndarray, concept_id: str) -> None:
        current = np.asarray(data)
        if len(current) == 0:
            raise ValueError(f"ReplayCAD cannot learn empty concept '{concept_id}'")

        try:
            replay = self._memory.generate_previous()
            self._memory.release_device_memory()

            train_data = current if len(replay) == 0 else np.concatenate([self._match_shape(replay, current), current])
            self._model.fit(train_data)

            self._memory.compress(concept_id=concept_id, images=current)
        finally:
            self._memory.release_device_memory()

    def predict(self, data: np.ndarray, concept_id: str) -> VisionPredictionResults:
        del concept_id  # ReplayCAD keeps no per-concept inference state
        return self._model.predict(data)

    def name(self) -> str:
        return "ReplayCAD"

    def additional_info(self) -> Dict[str, Any]:
        return {"model": self._model.name(), "replaycad": self._memory.info()}

    @staticmethod
    def _match_shape(replay: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Resize replay images to the current concept's spatial shape and channel count."""
        if replay.ndim != 4 or current.ndim != 4:
            raise ValueError(
                f"ReplayCAD expects (N, H, W, C) arrays; got replay={replay.shape}, current={current.shape}"
            )

        height, width, channels = current.shape[1:]
        if channels not in (1, 3):
            raise ValueError(f"ReplayCAD supports one or three image channels, got {channels}")

        resized = []
        for image in replay:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            pil = Image.fromarray(array[..., :3].astype(np.uint8))
            pil = pil.convert("L" if channels == 1 else "RGB")
            pil = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
            restored = np.asarray(pil)
            resized.append(restored[..., None] if channels == 1 else restored)

        return np.stack(resized).astype(current.dtype, copy=False)
