import numpy as np
import pytest

from pyclad.models.model import Model
from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.prediction_results import VisionPredictionResults


def test_vision_model_cannot_be_instantiated_without_predict():
    class IncompleteVisionModel(VisionModel):
        def fit(self, data: np.ndarray) -> None: ...

        def name(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteVisionModel()


def test_vision_model_is_a_model_subclass():
    assert issubclass(VisionModel, Model)


def test_complete_vision_model_can_be_instantiated():
    class _ConcreteVisionModel(VisionModel):
        def fit(self, data: np.ndarray) -> None: ...

        def predict(self, data: np.ndarray) -> VisionPredictionResults:
            return VisionPredictionResults(
                y_pred=np.zeros(len(data), dtype=np.int64),
                anomaly_scores=np.zeros(len(data), dtype=np.float32),
                score_maps=np.zeros((len(data), 4, 4), dtype=np.float32),
            )

        def name(self) -> str:
            return "concrete"

    model = _ConcreteVisionModel()
    assert isinstance(model, VisionModel)
    assert isinstance(model, Model)
