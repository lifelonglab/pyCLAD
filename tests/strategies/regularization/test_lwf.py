import numpy as np
import pytest
import torch
from torch import nn

from pyclad.models.adapters.temporal_adapter import FlattenTimeSeriesAdapter
from pyclad.models.autoencoder.autoencoder import Autoencoder, TemporalAutoencoder, VariationalTemporalAutoencoder
from pyclad.models.model import Model
from pyclad.strategies.regularization.lwf import LwFStrategy


class ScalarLinearModule(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.train_loss = nn.MSELoss()
        with torch.no_grad():
            self.linear.weight.fill_(weight)

    def forward(self, x):
        return self.linear(x)


class SimpleTorchModel(Model):
    def __init__(
        self,
        weight: float = 0.0,
        *,
        module: nn.Module | None = None,
        batch_size: int = 1,
        lr: float = 0.1,
        epochs: int = 1,
        fit_delta: float = 0.0,
    ):
        self.module = module if module is not None else ScalarLinearModule(weight)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.fit_delta = fit_delta
        self.fit_calls = 0

    def fit(self, data: np.ndarray):
        del data
        self.fit_calls += 1
        if self.fit_delta == 0.0:
            return
        with torch.no_grad():
            for parameter in self.module.parameters():
                if parameter.requires_grad:
                    parameter.add_(self.fit_delta)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "SimpleTorchModel"


class RecordingVariationalLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, x, x_hat, mean, var):
        self.calls.append((x.detach().clone(), x_hat.detach().clone(), mean.detach().clone(), var.detach().clone()))
        return torch.nn.functional.mse_loss(x_hat, x)


class VariationalLikeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(1.0))
        self.train_loss = RecordingVariationalLoss()

    def forward(self, x):
        return x + self.offset, x + 2 * self.offset, x + 3 * self.offset


class TupleEncoder(nn.Module):
    def forward(self, x):
        return x + 1.0, x + 2.0


class TupleEncoderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.encoder = TupleEncoder()
        self.decoder = nn.Identity()
        self.train_loss = nn.MSELoss()

    def forward(self, x):
        latent, _ = self.encoder(x * self.scale)
        return self.decoder(latent)


class StochasticVariationalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        mean = x * self.scale
        std = torch.ones_like(mean)
        return mean, std


class StochasticVariationalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StochasticVariationalEncoder()
        self.decoder = nn.Identity()
        self.train_loss = nn.MSELoss()

    def forward(self, x):
        mean, std = self.encoder(x)
        return self.decoder(mean + std * torch.randn_like(std)), mean, std


def test_distillation_loss_changes_training_update():
    data = np.array([[1.0]], dtype=np.float32)

    no_distill = LwFStrategy(SimpleTorchModel(weight=0.0, epochs=1), alpha=0.0, distill_mode="reconstruction")
    no_distill._old_model = SimpleTorchModel(weight=-1.0, epochs=1)
    no_distill._fit_with_distillation(data)

    with_distill = LwFStrategy(SimpleTorchModel(weight=0.0, epochs=1), alpha=3.0, distill_mode="reconstruction")
    with_distill._old_model = SimpleTorchModel(weight=-1.0, epochs=1)
    with_distill._fit_with_distillation(data)

    no_distill_weight = no_distill._model.module.linear.weight.item()
    with_distill_weight = with_distill._model.module.linear.weight.item()

    assert no_distill_weight > 0.0
    assert with_distill_weight < 0.0
    assert with_distill_weight != pytest.approx(no_distill_weight)


def test_training_batch_step_uses_module_train_loss_for_variational_outputs():
    model = SimpleTorchModel(module=VariationalLikeModule())
    strategy = LwFStrategy(model, distill_mode="reconstruction")
    batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    loss, x_hat, z, _ = strategy._training_batch_step(batch)

    assert loss.item() == pytest.approx(1.0)
    assert z is None
    assert torch.equal(x_hat, batch + 1.0)
    assert len(model.module.train_loss.calls) == 1
    seen_x, seen_x_hat, seen_mean, seen_var = model.module.train_loss.calls[0]
    assert torch.equal(seen_x, batch)
    assert torch.equal(seen_x_hat, batch + 1.0)
    assert torch.equal(seen_mean, batch + 2.0)
    assert torch.equal(seen_var, batch + 3.0)


def test_resolve_shuffle_matches_autoencoder_temporal_behavior():
    dense_strategy = LwFStrategy(Autoencoder(nn.Identity(), nn.Identity()), distill_mode="reconstruction")
    temporal_strategy = LwFStrategy(TemporalAutoencoder(nn.Identity(), nn.Identity()), distill_mode="reconstruction")
    wrapped_temporal_strategy = LwFStrategy(
        FlattenTimeSeriesAdapter(TemporalAutoencoder(nn.Identity(), nn.Identity())),
        distill_mode="reconstruction",
    )

    assert dense_strategy._resolve_shuffle() is True
    assert temporal_strategy._resolve_shuffle() is False
    assert wrapped_temporal_strategy._resolve_shuffle() is False


@pytest.mark.parametrize("distill_mode", ["latent", "hybrid"])
def test_latent_modes_accept_tuple_encoder_outputs(distill_mode):
    strategy = LwFStrategy(SimpleTorchModel(module=TupleEncoderModule()), distill_mode=distill_mode)
    strategy._old_model = strategy._clone_model()

    strategy._fit_with_distillation(np.array([[1.0]], dtype=np.float32))


def test_latent_distillation_accepts_variational_temporal_autoencoder_tuple_latents():
    model = VariationalTemporalAutoencoder(StochasticVariationalEncoder(), nn.Identity(), epochs=1)
    strategy = LwFStrategy(model, distill_mode="latent")
    strategy._old_model = strategy._clone_model()

    strategy._fit_with_distillation(np.ones((2, 2), dtype=np.float32))


def test_latent_mode_falls_back_to_reconstruction_without_encoder_hooks():
    strategy = LwFStrategy(SimpleTorchModel(weight=0.0), distill_mode="latent")
    data = np.array([[1.0]], dtype=np.float32)

    strategy.learn(data)
    strategy.learn(data)

    assert strategy._task_count == 2
    assert strategy._old_model is not None


def test_variational_reconstruction_distillation_uses_deterministic_mean_path():
    strategy = LwFStrategy(
        SimpleTorchModel(module=StochasticVariationalModule()),
        distill_mode="reconstruction",
    )
    strategy._old_model = strategy._clone_model()
    batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    stochastic_x_hat = batch + torch.randn_like(batch)

    distill_loss = strategy._compute_distill_loss(batch, "reconstruction", stochastic_x_hat, None)

    assert distill_loss.item() == pytest.approx(0.0)


def test_adaptive_balance_scales_distillation_relative_to_reconstruction_loss():
    strategy = LwFStrategy(SimpleTorchModel(weight=0.0), alpha=0.5)

    weighted_loss = strategy._weight_distill_loss(torch.tensor(2.0), torch.tensor(0.01))

    assert weighted_loss.item() == pytest.approx(1.0)


def test_strategy_epochs_override_controls_distillation_training(monkeypatch):
    model = SimpleTorchModel(weight=0.0, batch_size=2, epochs=1)
    strategy = LwFStrategy(model, epochs=3, distill_mode="reconstruction")
    strategy._old_model = strategy._clone_model()
    call_count = 0

    def counted_training_step(batch, *, include_latent=False):
        nonlocal call_count
        del include_latent
        call_count += 1
        x_hat = model.module(batch)
        rec_loss = sum(parameter.pow(2).mean() for parameter in model.module.parameters())
        return rec_loss, x_hat, None, None

    monkeypatch.setattr(strategy, "_training_batch_step", counted_training_step)
    monkeypatch.setattr(
        strategy,
        "_compute_distill_loss",
        lambda batch, distill_mode, x_hat_new, z_new: next(model.module.parameters()).sum() * 0.0,
    )

    strategy._fit_with_distillation(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32))

    assert call_count == 6
    assert strategy.additional_info()["epochs"] == 3


def test_strategy_epochs_override_must_be_positive():
    with pytest.raises(ValueError, match="epochs must be positive"):
        LwFStrategy(SimpleTorchModel(weight=0.0), epochs=0)


def test_learn_replaces_single_previous_task_teacher_snapshot():
    strategy = LwFStrategy(SimpleTorchModel(weight=0.0, epochs=2), alpha=0.5, distill_mode="reconstruction")
    data = np.array([[1.0]], dtype=np.float32)

    strategy.learn(data)
    first_teacher = strategy._old_model
    first_teacher_weight = first_teacher.module.linear.weight.detach().clone()

    strategy.learn(data)
    second_teacher = strategy._old_model
    current_weight = strategy._model.module.linear.weight.detach().clone()

    assert first_teacher is not second_teacher
    assert not torch.equal(first_teacher_weight, current_weight)
    assert torch.equal(second_teacher.module.linear.weight.detach(), current_weight)
    assert all(not parameter.requires_grad for parameter in second_teacher.module.parameters())
