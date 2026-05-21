"""Smoke test for the CARLA model.

Trains a tiny CARLA on synthetic windowed data and checks that fit + predict
run end-to-end with sane output shapes and ranges.
"""

import numpy as np
import pytest

from pyclad.models.backbones.resnet import ResNet1D, ResNetBlockConfig
from pyclad.models.contrastive.carla import Carla


_WINDOW_SIZE = 20
_NUM_FEATURES = 3
_N_WINDOWS = 32


@pytest.fixture(scope="module")
def windows() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, _WINDOW_SIZE, dtype=np.float32)
    base = np.sin(t)[None, :, None] * np.ones(
        (_N_WINDOWS, _WINDOW_SIZE, _NUM_FEATURES), dtype=np.float32
    )
    noise = (
        rng.standard_normal((_N_WINDOWS, _WINDOW_SIZE, _NUM_FEATURES)).astype(
            np.float32
        )
        * 0.05
    )
    return base + noise


@pytest.fixture(scope="module")
def trained_model(windows: np.ndarray) -> Carla:
    backbone = ResNet1D(
        num_features=_NUM_FEATURES,
        blocks=[
            ResNetBlockConfig(out_channels=4, kernel_sizes=[3, 3, 3]),
            ResNetBlockConfig(out_channels=8, kernel_sizes=[3, 3, 3]),
            ResNetBlockConfig(out_channels=8, kernel_sizes=[3, 3, 3]),
        ],
    )
    model = Carla(
        backbone,
        projection_dim=4,
        n_classes=3,
        num_neighbours=2,
        margin=1.0,
        entropy_loss_weight=2.0,
        positive_lookback=2,
        batch_size=8,
        pretext_epochs=2,
        classification_epochs=2,
        random_seed=0,
        device="cpu",
    )
    model.fit(windows)
    return model


def test_predict_shapes(trained_model: Carla, windows: np.ndarray) -> None:
    result = trained_model.predict(windows)
    assert result.y_pred.shape == (len(windows),)
    assert result.anomaly_scores.shape == (len(windows),)


def test_predict_labels_are_binary(trained_model: Carla, windows: np.ndarray) -> None:
    result = trained_model.predict(windows)
    assert set(np.unique(result.y_pred)).issubset({0, 1})


def test_predict_scores_are_finite_probabilities(
    trained_model: Carla, windows: np.ndarray
) -> None:
    result = trained_model.predict(windows)
    assert np.all(np.isfinite(result.anomaly_scores))
    assert np.all(result.anomaly_scores >= 0.0)
    assert np.all(result.anomaly_scores <= 1.0)
