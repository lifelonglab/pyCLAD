import numpy as np
import pytest

from pyclad.vision.models.patchcore.config import PatchCoreConfig
from pyclad.vision.models.patchcore.patchcore import PatchCore
from pyclad.vision.prediction_results import VisionPredictionResults


def _config(**overrides) -> PatchCoreConfig:
    defaults = dict(
        input_size=(32, 32),
        backbone_name="resnet18",
        pretrained_backbone=False,
        input_range="float01",
        batch_size=2,
        coreset_sampling_ratio=0.5,
        coreset_projection_dimension=8,
        coreset_starting_points=2,
        pretrain_embed_dimension=32,
        target_embed_dimension=32,
    )
    defaults.update(overrides)
    return PatchCoreConfig(**defaults)


def test_patchcore_fit_then_predict_shapes():
    rng = np.random.default_rng(0)
    train = rng.random((4, 32, 32, 3), dtype=np.float32)
    test = rng.random((3, 32, 32, 3), dtype=np.float32)

    model = PatchCore(_config())
    model.fit(train)
    result = model.predict(test)

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (3,)
    assert result.anomaly_scores.shape == (3,)
    assert result.score_maps.shape == (3, 32, 32)


def test_patchcore_predict_handles_empty_input():
    model = PatchCore(_config(threshold=0.0))

    result = model.predict(np.empty((0, 32, 32, 3), dtype=np.float32))

    assert result.y_pred.shape == (0,)
    assert result.score_maps.shape == (0,)


def test_patchcore_predict_before_fit_raises():
    model = PatchCore(_config())

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.zeros((1, 32, 32, 3), dtype=np.float32))


def test_image_score_is_max_of_raw_patch_scores_not_of_smoothed_map():
    rng = np.random.default_rng(3)
    train = rng.random((4, 32, 32, 3), dtype=np.float32)
    test = rng.random((2, 32, 32, 3), dtype=np.float32)

    model = PatchCore(_config(smoothing_sigma=8.0))
    model.fit(train)
    result = model.predict(test)

    # Heavy smoothing pulls the map maximum well below the raw patch maximum, so an
    # implementation that aggregated the returned map would report smaller scores.
    assert np.all(result.anomaly_scores > result.score_maps.reshape(2, -1).max(axis=1))


def test_patch_score_is_squared_l2_distance_like_the_reference_faiss_index():
    # ADer's reference NearestNeighbourScorer is backed by faiss.IndexFlatL2, which returns
    # SQUARED L2 distances. sklearn's NearestNeighbors.kneighbors returns plain (non-squared)
    # Euclidean distances, so PatchCore must square them itself to match. Pin this with a fully
    # deterministic embedding (bypassing the real backbone) rather than only checking shapes: a
    # memory bank containing the origin and a test patch at distance 3 along one axis must produce
    # an image score of 3**2 == 9, not 3 -- the value plain sklearn distances would give.
    model = PatchCore(_config(n_neighbors=1, coreset_sampling_ratio=1.0))

    def fake_train_embed(batch):
        return np.zeros((1, 4), dtype=np.float32), [[1, 1]]

    model._embed = fake_train_embed
    model.fit(np.zeros((1, 32, 32, 3), dtype=np.float32))

    def fake_test_embed(batch):
        return np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32), [[1, 1]]

    model._embed = fake_test_embed
    result = model.predict(np.zeros((1, 32, 32, 3), dtype=np.float32))

    assert result.anomaly_scores[0] == pytest.approx(9.0)
    assert result.anomaly_scores[0] != pytest.approx(3.0)  # plain (unsquared) Euclidean distance


def test_patchcore_seed_makes_coreset_reproducible():
    rng = np.random.default_rng(11)
    train = rng.random((8, 32, 32, 3), dtype=np.float32)
    test = rng.random((2, 32, 32, 3), dtype=np.float32)

    def scores(seed: int) -> np.ndarray:
        model = PatchCore(_config(seed=seed, coreset_sampling_ratio=0.3))
        model.fit(train)
        return model.predict(test).anomaly_scores

    assert np.array_equal(scores(5), scores(5))


def test_patchcore_requests_v1_weights_by_default():
    captured = {}

    import pyclad.vision.models.patchcore.patchcore as module

    original = module.TorchvisionFeatureExtractor

    class Recording(original):
        def __init__(self, *args, **kwargs):
            captured["weights_name"] = kwargs.get("weights_name")
            super().__init__(*args, **kwargs)

    module.TorchvisionFeatureExtractor = Recording
    try:
        PatchCore(_config())
    finally:
        module.TorchvisionFeatureExtractor = original

    assert captured["weights_name"] == "IMAGENET1K_V1"


def test_unknown_backbone_has_no_default_return_nodes():
    with pytest.raises(ValueError, match="return nodes"):
        PatchCore._require_nodes("definitely_not_a_backbone")


def test_explicit_return_nodes_win_over_the_defaults():
    model = PatchCore(_config(backbone_return_nodes=("layer1", "layer2")))

    assert model.module.return_nodes == ("layer1", "layer2")


def test_score_maps_match_input_spatial_size():
    rng = np.random.default_rng(1)
    train = rng.random((4, 48, 64, 3), dtype=np.float32)

    model = PatchCore(_config(input_size=(32, 32)))
    model.fit(train)
    result = model.predict(train[:2])

    # VisionScoringBase resizes maps back to each image's own spatial size.
    assert result.score_maps.shape == (2, 48, 64)
