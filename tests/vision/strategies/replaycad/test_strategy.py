import numpy as np
import pytest

from pyclad.vision.prediction_results import VisionPredictionResults
from pyclad.vision.strategies.replaycad.memory import ReplayCADMemory
from pyclad.vision.strategies.replaycad.strategy import ReplayCADStrategy


class RecordingModel:
    def __init__(self, input_size=(8, 8)):
        self.fit_calls: list[np.ndarray] = []
        self.input_size = input_size

    def fit(self, data: np.ndarray):
        self.fit_calls.append(np.asarray(data))

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        count = len(data)
        return VisionPredictionResults(
            y_pred=np.zeros(count, dtype=np.int64),
            anomaly_scores=np.zeros(count, dtype=np.float32),
            score_maps=np.zeros((count, *self.input_size), dtype=np.float32),
        )

    def name(self) -> str:
        return "RecordingModel"

    def additional_info(self) -> dict:
        return {}


def _images(count: int = 4, size: int = 8) -> np.ndarray:
    return np.random.default_rng(1).integers(0, 256, size=(count, size, size, 3), dtype=np.uint8)


def _strategy(config, backend, model=None):
    memory = ReplayCADMemory(config=config, backend=backend, benchmark="mvtec")
    return ReplayCADStrategy(model or RecordingModel(), memory), memory


def test_first_concept_trains_on_real_data_only(replaycad_config, stub_backend):
    config = replaycad_config()
    model = RecordingModel()
    strategy, _ = _strategy(config, stub_backend(config), model)
    data = _images(4)

    strategy.learn(data, concept_id="mvtec__screw")

    assert len(model.fit_calls) == 1
    assert np.array_equal(model.fit_calls[0], data)


def test_second_concept_trains_on_replay_then_new_data(replaycad_config, stub_backend):
    config = replaycad_config(replay_samples_per_concept=3)
    model = RecordingModel()
    strategy, _ = _strategy(config, stub_backend(config), model)

    strategy.learn(_images(4), concept_id="mvtec__screw")
    new_data = _images(2)
    strategy.learn(new_data, concept_id="mvtec__pill")

    combined = model.fit_calls[1]
    assert len(combined) == 3 + 2  # replay for screw, then pill's own images
    assert np.array_equal(combined[3:], new_data)


def test_replay_is_resized_to_the_new_concept_shape(replaycad_config, stub_backend):
    # The diffusion model generates at its own resolution; the detector needs one array.
    config = replaycad_config(resolution=8, replay_samples_per_concept=2)
    model = RecordingModel()
    strategy, _ = _strategy(config, stub_backend(config), model)

    strategy.learn(_images(2, size=8), concept_id="mvtec__screw")
    strategy.learn(_images(2, size=16), concept_id="mvtec__pill")

    assert model.fit_calls[1].shape == (4, 16, 16, 3)


def test_compression_runs_after_fitting(replaycad_config, stub_backend):
    config = replaycad_config()
    backend = stub_backend(config)
    order: list[str] = []

    class OrderingModel(RecordingModel):
        def fit(self, data):
            order.append("fit")
            super().fit(data)

    original_compress = backend.compress

    def tracking_compress(*args, **kwargs):
        order.append("compress")
        return original_compress(*args, **kwargs)

    backend.compress = tracking_compress
    strategy, _ = _strategy(config, backend, OrderingModel())

    strategy.learn(_images(4), concept_id="mvtec__screw")

    assert order == ["fit", "compress"]


def test_learn_follows_the_generate_release_fit_compress_release_order(replaycad_config, stub_backend):
    """Pins the position of each step, not just how many times each happened."""
    config = replaycad_config()
    backend = stub_backend(config)
    order: list[str] = []

    class OrderingModel(RecordingModel):
        def fit(self, data):
            order.append("fit")
            super().fit(data)

    for attribute, label in (("generate", "generate"), ("compress", "compress"), ("release_device_memory", "release")):
        original = getattr(backend, attribute)

        def record(*args, _original=original, _label=label, **kwargs):
            order.append(_label)
            return _original(*args, **kwargs)

        setattr(backend, attribute, record)

    strategy, _ = _strategy(config, backend, OrderingModel())
    strategy.learn(_images(4), concept_id="mvtec__screw")
    strategy.learn(_images(4), concept_id="mvtec__pill")

    assert order == [
        # First concept: nothing to replay, so the backend is never asked to generate.
        "release",
        "fit",
        "compress",
        "release",
        # Second concept: screw is replayed before the detector sees anything.
        "generate",
        "release",
        "fit",
        "compress",
        "release",
    ]


def test_device_memory_is_released_even_when_fitting_fails(replaycad_config, stub_backend):
    config = replaycad_config()
    backend = stub_backend(config)

    class FailingModel(RecordingModel):
        def fit(self, data):
            raise RuntimeError("simulated cuda out of memory")

    strategy, _ = _strategy(config, backend, FailingModel())

    with pytest.raises(RuntimeError, match="simulated cuda out of memory"):
        strategy.learn(_images(4), concept_id="mvtec__screw")

    # One release always happens before fit; a second is only possible via the finally running
    # despite fit's exception. ">= 1" would pass even without a finally, so this pins the count.
    assert backend.release_calls == 2


def test_empty_concept_is_rejected(replaycad_config, stub_backend):
    config = replaycad_config()
    strategy, _ = _strategy(config, stub_backend(config))

    with pytest.raises(ValueError, match="empty concept"):
        strategy.learn(np.zeros((0, 8, 8, 3), dtype=np.uint8), concept_id="mvtec__screw")


def test_predict_delegates_to_the_detector(replaycad_config, stub_backend):
    config = replaycad_config()
    strategy, _ = _strategy(config, stub_backend(config))

    result = strategy.predict(_images(3), concept_id="mvtec__screw")

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (3,)


def test_info_reports_the_mechanism(replaycad_config, stub_backend):
    config = replaycad_config()
    strategy, _ = _strategy(config, stub_backend(config))

    info = strategy.info()["strategy"]

    assert info["name"] == "ReplayCAD"
    assert info["model"] == "RecordingModel"
    assert info["replaycad"]["semantic_tokens"] == config.semantic_tokens
