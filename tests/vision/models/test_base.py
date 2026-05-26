import numpy as np
import pytest

from pyclad.models.model import Model
from pyclad.vision.models.vision_model import VisionModel


def test_vision_model_cannot_be_instantiated_without_score_maps():
    class IncompleteVisionModel(VisionModel):
        def fit(self, data: np.ndarray) -> None: ...

        def predict(self, data: np.ndarray):
            return np.array([]), np.array([])

        def name(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError, match="score_maps"):
        IncompleteVisionModel()


def test_vision_model_is_a_model_subclass():
    assert issubclass(VisionModel, Model)


def test_complete_vision_model_can_be_instantiated():
    class _ConcreteVisionModel(VisionModel):
        def fit(self, data: np.ndarray) -> None: ...

        def predict(self, data: np.ndarray):
            return np.array([]), np.array([])

        def name(self) -> str:
            return "concrete"

        def score_maps(self, data: np.ndarray) -> np.ndarray:
            return np.zeros((len(data), 4, 4), dtype=np.float32)

    model = _ConcreteVisionModel()
    assert isinstance(model, VisionModel)
    assert isinstance(model, Model)
