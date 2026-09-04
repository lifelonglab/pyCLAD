import numpy as np
import pytest

from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.ucad import UCAD
from pyclad.vision.prediction_results import VisionPredictionResults
from pyclad.vision.strategies.ucad.strategy import UCADStrategy

pytest.importorskip("timm")


def _model() -> UCAD:
    return UCAD(
        UCADConfig(
            backbone_name="vit_tiny_patch16_224",
            pretrained_backbone=False,
            input_size=(32, 32),
            input_range="float01",
            batch_size=2,
            feature_layer=3,
            prompt_depth=3,
            target_embed_dimension=16,
            key_size=3,
            knowledge_size=3,
            coreset_projection_dimension=4,
            coreset_starting_points=2,
            epochs=1,
            seed=0,
        )
    )


def _images(count: int, fill: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((count, 32, 32, 3), dtype=np.float32) * 0.1 + fill).astype(np.float32)


def test_learn_forwards_the_concept_id_to_the_model():
    strategy = UCADStrategy(_model())

    strategy.learn(_images(4, 0.2), concept_id="bottle")

    assert [memory.concept_id for memory in strategy._model.task_memories] == ["bottle"]


def test_learn_accumulates_tasks_across_the_stream():
    strategy = UCADStrategy(_model())

    strategy.learn(_images(4, 0.05, seed=1), concept_id="dark")
    strategy.learn(_images(4, 0.85, seed=2), concept_id="bright")

    assert len(strategy._model.task_memories) == 2
    assert strategy.additional_info()["tasks_seen"] == 2


def test_predict_ignores_the_concept_id_it_is_given():
    strategy = UCADStrategy(_model())
    strategy.learn(_images(4, 0.05, seed=1), concept_id="dark")
    strategy.learn(_images(4, 0.85, seed=2), concept_id="bright")
    data = _images(3, 0.05, seed=3)

    honest = strategy.predict(data, concept_id="dark")
    misleading = strategy.predict(data, concept_id="bright")
    absent = strategy.predict(data)

    np.testing.assert_array_equal(honest.anomaly_scores, misleading.anomaly_scores)
    np.testing.assert_array_equal(honest.anomaly_scores, absent.anomaly_scores)
    assert isinstance(honest, VisionPredictionResults)


def test_strategy_reports_itself():
    strategy = UCADStrategy(_model())

    assert strategy.name() == "UCAD"
    assert strategy.additional_info()["task_agnostic_inference"] is True
