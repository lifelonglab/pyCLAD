import numpy as np
import pytest

from pyclad.models.model import Model
from pyclad.models.neural_model import NeuralModel
from pyclad.models.autoencoder.continual_ae1svm import ContinualAE1SVM
from pyclad.models.autoencoder.continual_deep_svdd import ContinualDeepSVDD
from pyclad.models.autoencoder.continual_vae import ContinualVAE
from pyclad.strategies.regularization.ewc import EWCStrategy
from pyclad.strategies.regularization.lwf import LwFStrategy
from pyclad.strategies.replay.agem import AGEMStrategy
from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer


class ClassicalModel(Model):
    def fit(self, data):
        pass

    def predict(self, data):
        return np.zeros(len(data), dtype=np.int64), np.zeros(len(data), dtype=np.float32)

    def name(self):
        return "ClassicalModel"


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


def _vae():
    return ContinualVAE(
        encoder_neuron_list=[4],
        decoder_neuron_list=[4],
        latent_dim=2,
        preprocessing=False,
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


def _deep_svdd(use_ae=False):
    return ContinualDeepSVDD(
        n_features=3,
        hidden_neurons=[4, 2],
        preprocessing=False,
        use_ae=use_ae,
        verbose=0,
        epochs=1,
        batch_size=2,
    )


def test_neural_model_is_a_model_subclass():
    assert issubclass(NeuralModel, Model)


@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda model: AGEMStrategy(
            model,
            BalancedReplayBuffer(max_size=8, seed=1),
            batch_size=2,
            replay_batch_size=2,
            epochs=1,
            lr=1e-3,
            device="cpu",
        ),
        lambda model: EWCStrategy(model, ewc_lambda=0.1),
        lambda model: LwFStrategy(model, alpha=0.1, distill_mode="reconstruction"),
    ],
)
def test_gradient_strategies_reject_classical_models(strategy_factory):
    with pytest.raises(TypeError, match="requires a neural model"):
        strategy_factory(ClassicalModel())


@pytest.mark.parametrize("model_factory", [_vae, _ae1svm, _deep_svdd])
def test_ewc_builds_uninitialized_neural_models_and_predicts_scores(model_factory):
    model = model_factory()
    assert model.module_getter() is None

    strategy = EWCStrategy(model, ewc_lambda=0.1)
    strategy.learn(_small_data())
    labels, scores = strategy.predict(_small_data(0.05))

    assert model.module_getter() is not None
    assert labels.shape == (4,)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert strategy._task_count == 1


@pytest.mark.parametrize("model_factory", [_vae, _ae1svm, _deep_svdd])
def test_lwf_runs_two_tasks_with_neural_model_hooks(model_factory):
    model = model_factory()
    strategy = LwFStrategy(model, alpha=0.1, distill_mode="latent")

    strategy.learn(_small_data(), concept_id="first")
    strategy.learn(_small_data(0.05), concept_id="second")
    labels, scores = strategy.predict(_small_data(0.10))

    assert labels.shape == (4,)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert strategy._task_count == 2
    assert strategy._old_model is not None


@pytest.mark.parametrize("use_ae", [False, True])
def test_continual_deep_svdd_fit_completes(use_ae):
    model = _deep_svdd(use_ae=use_ae)
    model.fit(_small_data())
    labels, scores = model.predict(_small_data(0.05))

    assert labels.shape == (4,)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
