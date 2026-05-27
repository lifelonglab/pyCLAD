"""Integration test for the vision scenario + pixel callback pipeline.

The base ConceptIncrementalScenario now unpacks VisionPredictionResults and
passes score_maps to callbacks automatically — no separate vision scenario needed.
"""

import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.output.prediction_results import PredictionResults
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.prediction_results import VisionPredictionResults


class _PixelPerfectModel(VisionModel):
    """Returns score_maps that exactly match the ground-truth masks."""

    def fit(self, data: np.ndarray) -> None: ...

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        return VisionPredictionResults(
            y_pred=np.zeros(len(data), dtype=np.int64),
            anomaly_scores=np.zeros(len(data), dtype=np.float32),
            score_maps=np.array(
                [
                    [[0.95, 0.05], [0.05, 0.05]],  # anomaly at (0,0)
                    [[0.05, 0.05], [0.05, 0.05]],  # no anomaly
                ],
                dtype=np.float32,
            ),
        )

    def name(self) -> str:
        return "PixelPerfect"


class _PlainModel:
    """Non-vision model — predict() returns plain PredictionResults."""

    def fit(self, data): ...

    def predict(self, data) -> PredictionResults:
        return PredictionResults(
            y_pred=np.zeros(len(data), dtype=np.int64),
            anomaly_scores=np.zeros(len(data), dtype=np.float32),
        )

    def name(self):
        return "Plain"


class _SingleModelStrategy(ConceptIncrementalStrategy):
    def __init__(self, model):
        self._model = model

    def learn(self, data: np.ndarray) -> None:
        self._model.fit(data)

    def predict(self, data: np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "SingleModelStrategy"


def _build_dataset_with_one_vision_concept() -> ConceptsDataset:
    train = Concept(name="widget", data=np.zeros((1, 2, 2, 3), dtype=np.float32))
    test = VisionConcept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
        masks=np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    )
    return ConceptsDataset(name="fixture", train_concepts=[train], test_concepts=[test])


def test_scenario_pipes_score_maps_to_pixel_callback():
    """VisionPredictionResults from the model flows through the base scenario to the callback."""
    dataset = _build_dataset_with_one_vision_concept()
    strategy = _SingleModelStrategy(_PixelPerfectModel())
    callback = VisionPixelConceptMetricCallback(base_metric=PixelRocAuc())

    ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[callback]).run()

    info = callback.info()["pixel_concept_metric_callback_Pixel-ROC-AUC"]
    assert info["concepts_order"] == ["widget"]
    assert info["metric_matrix"]["widget"]["widget"] == 1.0


def test_scenario_skips_pixel_callback_when_model_is_not_vision_model():
    """Plain PredictionResults (no score_maps) → pixel callback silently skips."""
    dataset = _build_dataset_with_one_vision_concept()
    strategy = _SingleModelStrategy(_PlainModel())
    callback = VisionPixelConceptMetricCallback(base_metric=PixelRocAuc())

    ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[callback]).run()

    assert callback.info() == {}
