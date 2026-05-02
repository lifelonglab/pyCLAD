import logging

import numpy as np
import pytest
import torch
from torch import nn

from pyclad.models.model import Model
from pyclad.strategies.regularization.ewc import EWCStrategy


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class TinyTorchModel(Model):
    def __init__(self):
        self.module = TinyModule()
        self.batch_size = 2
        self.lr = 1e-2
        self.epochs = 1
        self.device = torch.device("cpu")

    def fit(self, data: np.ndarray):
        raise AssertionError("EWCStrategy should not delegate training to model.fit().")

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=float)

    def name(self) -> str:
        return "TinyTorchModel"


class RecordingVariationalLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, x, x_hat, mean, var):
        self.calls.append((x.detach().clone(), x_hat.detach().clone(), mean.detach().clone(), var.detach().clone()))
        return torch.nn.functional.mse_loss(x_hat, x)


class VariationalLikeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_loss = RecordingVariationalLoss()

    def forward(self, x):
        return x + 1.0, x + 2.0, x + 3.0


class InternalTypeErrorLoss:
    def __call__(self, x_hat, x):
        raise TypeError("boom")


class ErroringLossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_loss = InternalTypeErrorLoss()

    def forward(self, x):
        return x


class AmbiguousLoss:
    def __call__(self, left, right):
        raise AssertionError("Ambiguous train_loss should not be invoked.")


class AmbiguousLossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_loss = AmbiguousLoss()

    def forward(self, x):
        return x + 1.0


class ExplodingTo:
    def to(self, *args, **kwargs):
        raise AssertionError("Penalty computation should use the prebuilt cache, not the saved CPU tensors.")


def test_ewc_strategy_default_lambda_is_conservative_starting_point():
    strategy = EWCStrategy(TinyTorchModel())

    assert strategy.additional_info()["ewc_lambda"] == pytest.approx(1.0)
    assert strategy.additional_info()["epochs"] == 1


def test_ewc_strategy_accepts_explicit_epochs_override():
    strategy = EWCStrategy(TinyTorchModel(), epochs=3)

    assert strategy.additional_info()["epochs"] == 3


def test_ewc_strategy_rejects_non_positive_epochs():
    with pytest.raises(ValueError, match="epochs must be positive"):
        EWCStrategy(TinyTorchModel(), epochs=0)


def test_prepare_data_reuses_identity_transform_storage():
    strategy = EWCStrategy(TinyTorchModel())
    data = np.array([[1.0, 2.0]], dtype=np.float32)

    prepared = strategy._prepare_data(data)

    assert np.shares_memory(prepared.numpy(), data)


def test_learn_prepares_tensor_data_once_and_reuses_it_for_fisher(monkeypatch):
    strategy = EWCStrategy(TinyTorchModel())
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    original_prepare_data = strategy._prepare_data
    prepare_call_count = 0
    prepared_for_fit = []
    prepared_for_fisher = []

    def counted_prepare_data(raw_data):
        nonlocal prepare_call_count
        prepare_call_count += 1
        return original_prepare_data(raw_data)

    def fake_fit(tensor_data, calibration_data):
        prepared_for_fit.append(tensor_data)
        assert isinstance(calibration_data, np.ndarray)

    def fake_fisher(tensor_data):
        prepared_for_fisher.append(tensor_data)
        return {}

    monkeypatch.setattr(strategy, "_prepare_data", counted_prepare_data)
    monkeypatch.setattr(strategy, "_fit_with_ewc", fake_fit)
    monkeypatch.setattr(strategy, "_compute_fisher_information", fake_fisher)

    strategy.learn(data)

    assert prepare_call_count == 1
    assert len(prepared_for_fit) == 1
    assert len(prepared_for_fisher) == 1
    assert prepared_for_fit[0] is prepared_for_fisher[0]


def test_default_loss_fn_resolves_variational_loss_argument_order():
    module = VariationalLikeModule()
    batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    loss = EWCStrategy.default_loss_fn(module, batch)

    assert loss.item() == pytest.approx(1.0)
    assert len(module.train_loss.calls) == 1
    seen_x, seen_x_hat, seen_mean, seen_var = module.train_loss.calls[0]
    assert torch.equal(seen_x, batch)
    assert torch.equal(seen_x_hat, batch + 1.0)
    assert torch.equal(seen_mean, batch + 2.0)
    assert torch.equal(seen_var, batch + 3.0)


def test_default_loss_fn_does_not_swallow_internal_type_errors():
    batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    with pytest.raises(TypeError, match="boom"):
        EWCStrategy.default_loss_fn(ErroringLossModule(), batch)


def test_default_loss_fn_logs_when_it_falls_back_to_mse(caplog):
    batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    with caplog.at_level(logging.DEBUG):
        loss = EWCStrategy.default_loss_fn(AmbiguousLossModule(), batch)

    assert loss.item() == pytest.approx(1.0)
    assert "Falling back to plain MSE" in caplog.text


def test_compute_ewc_penalty_uses_prebuilt_cache_not_saved_cpu_tensors():
    model = TinyTorchModel()
    strategy = EWCStrategy(model)

    with torch.no_grad():
        model.module.linear.weight.fill_(1.0)
        model.module.linear.bias.fill_(1.0)

    strategy._saved_params = {
        0: {name: torch.zeros_like(param.detach()).cpu() for name, param in model.module.named_parameters() if param.requires_grad}
    }
    strategy._importances = {
        0: {name: torch.ones_like(param.detach()).cpu() for name, param in model.module.named_parameters() if param.requires_grad}
    }
    strategy._rebuild_penalty_cache(device=torch.device("cpu"))

    strategy._saved_params = {0: {name: ExplodingTo() for name in strategy._penalty_importance_sum}}
    strategy._importances = {0: {name: ExplodingTo() for name in strategy._penalty_importance_sum}}

    penalty = strategy._compute_ewc_penalty(model.module)

    assert penalty.item() > 0.0


def test_compute_fisher_information_uses_per_sample_batches_and_logs_cost(caplog, monkeypatch):
    strategy = EWCStrategy(TinyTorchModel())
    batch_sizes = []
    original_compute_batch_loss = strategy._compute_batch_loss

    def record_batch_size(module, batch):
        batch_sizes.append(batch.shape[0])
        return original_compute_batch_loss(module, batch)

    monkeypatch.setattr(strategy, "_compute_batch_loss", record_batch_size)
    tensor_data = strategy._prepare_data(np.array([[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32))

    with caplog.at_level(logging.INFO):
        fisher = strategy._compute_fisher_information(tensor_data)

    assert batch_sizes == [1, 1, 1]
    assert "per-sample gradients" in caplog.text
    assert fisher


def test_fit_with_ewc_uses_strategy_epochs_override(monkeypatch):
    strategy = EWCStrategy(TinyTorchModel(), epochs=3)
    call_count = 0

    def constant_loss(module, batch):
        nonlocal call_count
        call_count += 1
        return sum(parameter.pow(2).mean() for parameter in module.parameters())

    monkeypatch.setattr(strategy, "_loss_fn", constant_loss)
    monkeypatch.setattr(strategy, "_compute_ewc_penalty", lambda module: next(module.parameters()).new_tensor(0.0))

    tensor_data = strategy._prepare_data(np.array([[1.0, 2.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=np.float32))

    strategy._fit_with_ewc(tensor_data, tensor_data.numpy())

    assert call_count == 6
