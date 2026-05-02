import logging

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from pyclad.models.neural_model import NeuralModel
from pyclad.models.autoencoder.continual_ae1svm import ContinualAE1SVM
from pyclad.models.autoencoder.continual_deep_svdd import ContinualDeepSVDD
from pyclad.models.autoencoder.continual_vae import ContinualVAE
from pyclad.strategies.neural_hooks import NeuralStrategyHooks
from pyclad.strategies.replay.agem import AGEMStrategy
from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer


class HookedModel:
    def __init__(self):
        self.module = None
        self.prepare_calls = []
        self.prepare_batch_calls = 0
        self.loss_calls = 0
        self.after_fit_data = None

    def prepare_fit(self, current_data, replay_data=None):
        self.prepare_calls.append(
            (
                np.array(current_data, copy=True),
                np.array(replay_data, copy=True),
            )
        )
        if self.module is None:
            self.module = torch.nn.Linear(current_data.shape[1], current_data.shape[1], bias=False)

    def module_getter(self):
        return self.module

    def prepare_data(self, data):
        self.prepare_batch_calls += 1
        return torch.as_tensor(data, dtype=torch.float32)

    def compute_loss(self, module, batch):
        self.loss_calls += 1
        return torch.nn.functional.mse_loss(module(batch), batch)

    def after_fit(self, data):
        self.after_fit_data = np.array(data, copy=True)

    def predict(self, data):
        return np.zeros(len(data), dtype=np.int64)

    def decision_function(self, data):
        return np.zeros(len(data), dtype=np.float32)

    def name(self):
        return "HookedModel"


class SimpleAutoencoderModel:
    def __init__(self):
        self.module = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 3),
        )

    def predict(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.module.eval()
        with torch.no_grad():
            batch = torch.as_tensor(data, dtype=torch.float32)
            reconstruction = self.module(batch).detach().numpy()
        scores = ((data - reconstruction) ** 2).mean(axis=1)
        return (scores > 0.5).astype(np.int64), scores

    def name(self):
        return "SimpleAutoencoderModel"


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


def test_prepare_fit_receives_replay_and_model_loss_is_used():
    model = HookedModel()
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=1),
        batch_size=2,
        replay_batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )

    first = _small_data()
    second = _small_data(0.1)
    strategy.learn(first, concept_id="first")
    strategy.learn(second, concept_id="second")

    assert len(model.prepare_calls) == 2
    assert_array_equal(model.prepare_calls[0][1], np.empty((0,), dtype=np.float32))
    assert_array_equal(model.prepare_calls[1][1], first)
    assert model.prepare_batch_calls > 0
    assert model.loss_calls > 0
    assert model.after_fit_data.shape == (8, 3)


def test_explicit_loss_fn_overrides_model_loss():
    model = HookedModel()
    explicit_loss_calls = []

    def explicit_loss_fn(module, batch):
        explicit_loss_calls.append(batch.shape[0])
        return torch.nn.functional.mse_loss(module(batch), batch)

    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=1),
        loss_fn=explicit_loss_fn,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )

    strategy.learn(_small_data(), concept_id="first")

    assert explicit_loss_calls
    assert model.loss_calls == 0


def test_agem_delegates_model_hooks_through_neural_strategy_hooks(monkeypatch):
    model = HookedModel()
    prepare_calls = []
    loss_calls = []
    original_prepare_fit = NeuralStrategyHooks.prepare_fit
    original_batch_loss = NeuralStrategyHooks.batch_loss

    def wrapped_prepare_fit(self, data, replay_data=None, model=None):
        prepare_calls.append(np.array(data, copy=True))
        return original_prepare_fit(self, data, replay_data=replay_data, model=model)

    def wrapped_batch_loss(self, module, batch, model=None, explicit_loss_fn=None):
        loss_calls.append(tuple(batch.shape))
        return original_batch_loss(self, module, batch, model=model, explicit_loss_fn=explicit_loss_fn)

    monkeypatch.setattr(NeuralStrategyHooks, "prepare_fit", wrapped_prepare_fit)
    monkeypatch.setattr(NeuralStrategyHooks, "batch_loss", wrapped_batch_loss)

    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=1),
        batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )

    strategy.learn(_small_data(), concept_id="first")

    assert prepare_calls
    assert loss_calls
    assert model.prepare_calls
    assert model.loss_calls > 0


def test_simple_autoencoder_uses_generic_fallbacks():
    model = SimpleAutoencoderModel()
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=1),
        batch_size=2,
        replay_batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )

    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(_small_data(0.1), concept_id="second")
    labels, scores = strategy.predict(_small_data(0.2))

    assert labels.shape[0] == 4
    assert scores.shape[0] == 4
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "model",
    [
        ContinualVAE(
            encoder_neuron_list=[4],
            decoder_neuron_list=[4],
            latent_dim=2,
            preprocessing=False,
            batch_size=2,
            device="cpu",
            verbose=0,
        ),
        ContinualAE1SVM(
            hidden_neurons=[4],
            kernel_approx_features=8,
            batch_norm=False,
            preprocessing=False,
        ),
        ContinualDeepSVDD(
            n_features=3,
            hidden_neurons=[4, 2],
            preprocessing=False,
            use_ae=False,
            verbose=0,
        ),
        ContinualDeepSVDD(
            n_features=3,
            hidden_neurons=[4, 2],
            preprocessing=False,
            use_ae=True,
            verbose=0,
        ),
    ],
)
def test_smoke_runs_with_model_owned_integrations(model):
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=2),
        batch_size=2,
        replay_batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )

    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(_small_data(0.1), concept_id="second")
    labels, scores = strategy.predict(_small_data(0.2))

    assert labels.shape[0] == 4
    assert scores.shape[0] == 4
    assert np.isfinite(scores).all()


def test_preprocessing_uses_current_and_replay_data_without_non_finite_gradients(caplog):
    model = ContinualVAE(
        encoder_neuron_list=[4],
        decoder_neuron_list=[4],
        latent_dim=2,
        preprocessing=True,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=8, seed=3),
        batch_size=2,
        replay_batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )
    replay_concept = np.asarray(
        [
            [0.0, 0.20, 0.10],
            [0.1, 0.30, 0.20],
            [0.2, 0.40, 0.30],
            [0.3, 0.50, 0.40],
        ],
        dtype=np.float32,
    )
    zero_variance_current = np.asarray(
        [
            [0.0, 0.0, 0.10],
            [0.1, 0.0, 0.20],
            [0.2, 0.0, 0.30],
            [0.3, 0.0, 0.40],
        ],
        dtype=np.float32,
    )

    strategy.learn(replay_concept, concept_id="first")
    with caplog.at_level(logging.WARNING, logger="pyclad.strategies.replay.agem"):
        strategy.learn(zero_variance_current, concept_id="second")

    prepared_replay = model.prepare_data(replay_concept).numpy()
    assert np.isfinite(prepared_replay).all()
    assert "non-finite" not in caplog.text


def test_ae1svm_batch_norm_handles_singleton_current_and_replay_batches():
    model = ContinualAE1SVM(
        hidden_neurons=[4],
        kernel_approx_features=8,
        batch_norm=True,
        preprocessing=False,
    )
    strategy = AGEMStrategy(
        model,
        BalancedReplayBuffer(max_size=4, seed=4),
        batch_size=2,
        replay_batch_size=1,
        epochs=1,
        lr=1e-3,
        device="cpu",
        shuffle=False,
    )

    strategy.learn(np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32), concept_id="singleton")
    strategy.learn(_small_data(), concept_id="full")
    labels, scores = strategy.predict(_small_data(0.1))

    assert labels.shape[0] == 4
    assert scores.shape[0] == 4
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "model",
    [
        ContinualVAE(
            encoder_neuron_list=[4],
            decoder_neuron_list=[4],
            latent_dim=2,
            preprocessing=False,
            batch_size=2,
            device="cpu",
            verbose=0,
        ),
        ContinualAE1SVM(
            hidden_neurons=[4],
            kernel_approx_features=8,
            batch_norm=False,
            preprocessing=False,
        ),
        ContinualDeepSVDD(
            n_features=3,
            hidden_neurons=[4, 2],
            preprocessing=False,
            use_ae=False,
            verbose=0,
        ),
    ],
)
def test_fit_calls_prepare_fit_before_training(model, monkeypatch):
    calls = []
    original_prepare_fit = model.prepare_fit

    def wrapped_prepare_fit(X, replay_data=None, y=None):
        calls.append("prepare_fit")
        return original_prepare_fit(X, replay_data=replay_data, y=y)

    def wrapped_train_fit_loader(train_loader):
        calls.append("train_fit_loader")
        assert len(train_loader) > 0

    def wrapped_after_fit(X):
        calls.append("after_fit")
        return model

    monkeypatch.setattr(model, "prepare_fit", wrapped_prepare_fit)
    monkeypatch.setattr(model, "train_fit_loader", wrapped_train_fit_loader)
    monkeypatch.setattr(model, "after_fit", wrapped_after_fit)

    assert model.fit(_small_data()) is model
    assert calls == ["prepare_fit", "train_fit_loader", "after_fit"]


def test_continual_model_classes_define_generic_hooks():
    inherited_hooks = {"module_getter", "prepare_fit", "prepare_data", "after_fit"}
    expected_hooks = {
        ContinualVAE: {"module_getter", "prepare_fit", "prepare_data", "compute_loss", "after_fit"},
        ContinualAE1SVM: {
            "module_getter",
            "prepare_fit",
            "prepare_data",
            "compute_loss",
            "after_fit",
            "drop_last",
        },
        ContinualDeepSVDD: {"module_getter", "prepare_fit", "prepare_data", "compute_loss", "after_fit"},
    }
    for model_class, hooks in expected_hooks.items():
        assert hooks.issubset(set(dir(model_class)))
        for hook in inherited_hooks:
            assert getattr(model_class, hook) is getattr(NeuralModel, hook)
