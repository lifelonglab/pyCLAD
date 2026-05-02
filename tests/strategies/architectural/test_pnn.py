import numpy as np
import pytest
from torch import nn

from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.models.autoencoder.continual_ae1svm import ContinualAE1SVM
from pyclad.models.autoencoder.continual_deep_svdd import ContinualDeepSVDD
from pyclad.models.autoencoder.continual_vae import ContinualVAE
from pyclad.strategies.architectural.pnn import PNNStrategy


def _small_data(offset=0.0):
    return np.asarray(
        [
            [0.05 + offset, 0.10, 0.20],
            [0.15 + offset, 0.20, 0.30],
            [0.25 + offset, 0.30, 0.40],
            [0.35 + offset, 0.40, 0.50],
        ],
        dtype=np.float32,
    )


def _autoencoder():
    return Autoencoder(
        nn.Sequential(nn.Linear(3, 2), nn.ReLU()),
        nn.Linear(2, 3),
        lr=1e-3,
        threshold=0.2,
        epochs=1,
        device="cpu",
    )


def _vae(preprocessing=False):
    return ContinualVAE(
        encoder_neuron_list=[4],
        decoder_neuron_list=[4],
        latent_dim=2,
        preprocessing=preprocessing,
        batch_size=2,
        epoch_num=1,
        device="cpu",
        verbose=0,
    )


def _ae1svm():
    return ContinualAE1SVM(
        hidden_neurons=[4],
        kernel_approx_features=8,
        batch_norm=False,
        preprocessing=False,
        epochs=1,
        batch_size=2,
    )


def _deep_svdd(use_ae=True):
    return ContinualDeepSVDD(
        n_features=3,
        hidden_neurons=[4, 2],
        preprocessing=False,
        use_ae=use_ae,
        verbose=0,
        epochs=1,
        batch_size=2,
    )


def _assert_finite_predictions(strategy, data, concept_id=None):
    labels, scores = strategy.predict(data, concept_id=concept_id)
    assert labels.shape == (len(data),)
    assert scores.shape == (len(data),)
    assert np.isfinite(scores).all()


def test_pnn_still_supports_existing_lightning_autoencoder_factory():
    strategy = PNNStrategy(_autoencoder, device="cpu")
    assert strategy.num_columns == 1

    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(_small_data(0.05), concept_id="second")

    assert strategy.num_columns == 2
    assert strategy._concept_to_task == {"first": 0, "second": 1}
    _assert_finite_predictions(strategy, _small_data(0.10), concept_id="second")


@pytest.mark.parametrize("model_factory", [_vae, _ae1svm, lambda: _deep_svdd(use_ae=True)])
def test_pnn_trains_two_concepts_with_lazy_neural_models(model_factory):
    strategy = PNNStrategy(model_factory, device="cpu")
    assert strategy.num_columns == 0

    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(_small_data(0.05), concept_id="second")

    assert strategy.num_columns == 2
    assert strategy._concept_to_task == {"first": 0, "second": 1}
    assert strategy.threshold is not None
    _assert_finite_predictions(strategy, _small_data(0.10), concept_id="second")


def test_pnn_rejects_non_reconstruction_deep_svdd():
    strategy = PNNStrategy(lambda: _deep_svdd(use_ae=False), device="cpu")

    with pytest.raises(ValueError, match="use_ae=True"):
        strategy.fit(_small_data())


def test_pnn_concept_prediction_uses_lazy_created_task_mapping():
    strategy = PNNStrategy(_vae, device="cpu")
    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(_small_data(0.05), concept_id="second")

    labels_by_concept, scores_by_concept = strategy.predict(_small_data(0.02), concept_id="first")
    labels_by_task, scores_by_task = strategy.predict(_small_data(0.02), task_label=0)

    np.testing.assert_array_equal(labels_by_concept, labels_by_task)
    np.testing.assert_allclose(scores_by_concept, scores_by_task)


def test_pnn_vae_preprocessing_context_scores_finite_values():
    strategy = PNNStrategy(lambda: _vae(preprocessing=True), device="cpu")
    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(
        np.asarray(
            [
                [0.00, 0.00, 0.10],
                [0.10, 0.00, 0.20],
                [0.20, 0.00, 0.30],
                [0.30, 0.00, 0.40],
            ],
            dtype=np.float32,
        ),
        concept_id="second",
    )

    _assert_finite_predictions(strategy, _small_data(0.05), concept_id="second")
