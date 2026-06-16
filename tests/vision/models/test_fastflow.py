import numpy as np
import pytest

from pyclad.vision.models.fastflow.config import FastFlowConfig
from pyclad.vision.models.fastflow.fastflow import FastFlow
from pyclad.vision.prediction_results import VisionPredictionResults


def _config(**overrides) -> FastFlowConfig:
    defaults = dict(
        input_size=(64, 64),
        backbone_name="resnet18",
        pretrained_backbone=False,
        flow_steps=2,
        input_range="float01",
        batch_size=1,
        epochs=0,
        show_training_progress=False,
    )
    defaults.update(overrides)
    return FastFlowConfig(**defaults)


@pytest.mark.parametrize("backbone_name", ["resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_fastflow_predict_shapes(backbone_name: str):
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(backbone_name=backbone_name, threshold=0.0))

    result = model.predict(data)

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (1,)
    assert result.anomaly_scores.shape == (1,)
    assert result.score_maps.shape == (1, 64, 64)


def test_fastflow_rejects_empty_return_nodes():
    with pytest.raises(ValueError, match="at least one feature node"):
        FastFlow(_config(backbone_return_nodes=()))


def test_fastflow_predict_handles_empty_input():
    model = FastFlow(_config(threshold=0.0))

    result = model.predict(np.empty((0, 64, 64, 3), dtype=np.float32))

    assert result.y_pred.shape == (0,)
    assert result.anomaly_scores.shape == (0,)
    assert result.score_maps.shape == (0,)


def test_fastflow_seed_makes_results_reproducible():
    data = np.random.default_rng(7).random((2, 64, 64, 3), dtype=np.float32)

    def score_maps(seed: int) -> np.ndarray:
        # pretrained_backbone=False -> backbone init is random; combined with the random
        # flow weights and channel permutations, the whole model is seed-dependent.
        model = FastFlow(_config(seed=seed, threshold=0.0))
        return model.predict(data).score_maps

    assert np.array_equal(score_maps(123), score_maps(123))  # same seed -> identical
    assert not np.array_equal(score_maps(123), score_maps(456))  # different seed -> different


def test_fastflow_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(batch_size=2, epochs=1))

    model.fit(data)
    result = model.predict(data)

    assert result.y_pred.shape == (2,)
    assert result.anomaly_scores.shape == (2,)
    assert result.score_maps.shape == (2, 64, 64)
    assert model.additional_info()["last_loss"] is not None
