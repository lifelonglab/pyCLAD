"""Integration test for the vision scenario + pixel callback pipeline.

Confirms the Opcja-2 contract: scenario fetches score_maps from the model
via Strategy.model_for_concept (no callback-side reflection) and forwards
them to callbacks through after_evaluation kwarg.
"""

import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.scenarios.concept_incremental import VisionConceptIncrementalScenario


class _PixelPerfectModel(VisionModel):
    """Returns score_maps that exactly match the ground-truth masks."""

    def fit(self, data: np.ndarray) -> None: ...

    def predict(self, data: np.ndarray):
        return np.zeros(len(data), dtype=np.int64), np.zeros(len(data), dtype=np.float32)

    def name(self) -> str:
        return "PixelPerfect"

    def score_maps(self, data: np.ndarray) -> np.ndarray:
        # encoded perfect predictions for the test fixture below
        return np.array(
            [
                [[0.95, 0.05], [0.05, 0.05]],  # anomaly at (0,0)
                [[0.05, 0.05], [0.05, 0.05]],  # no anomaly
            ],
            dtype=np.float32,
        )


class _SingleModelStrategy(ConceptIncrementalStrategy):
    """Minimal ConceptIncrementalStrategy storing a single shared model.
    Default Strategy.model_for_concept(self._model fallback) covers it."""

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
    dataset = _build_dataset_with_one_vision_concept()
    strategy = _SingleModelStrategy(_PixelPerfectModel())
    callback = VisionPixelConceptMetricCallback(base_metric=PixelRocAuc())

    scenario = VisionConceptIncrementalScenario(
        dataset=dataset,
        strategy=strategy,
        callbacks=[callback],
    )
    scenario.run()

    info = callback.info()["pixel_concept_metric_callback_Pixel-ROC-AUC"]
    assert info["concepts_order"] == ["widget"]
    assert info["metric_matrix"]["widget"]["widget"] == 1.0


def test_scenario_skips_pixel_callback_when_model_is_not_vision_model():
    """If the user runs a non-vision model under the vision scenario, the
    pixel callback should produce no output instead of erroring."""

    class _PlainModel:
        def fit(self, data): ...

        def predict(self, data):
            return np.zeros(len(data)), np.zeros(len(data))

        def name(self):
            return "Plain"

    dataset = _build_dataset_with_one_vision_concept()
    strategy = _SingleModelStrategy(_PlainModel())
    callback = VisionPixelConceptMetricCallback(base_metric=PixelRocAuc())

    scenario = VisionConceptIncrementalScenario(
        dataset=dataset,
        strategy=strategy,
        callbacks=[callback],
    )
    scenario.run()

    assert callback.info() == {}
