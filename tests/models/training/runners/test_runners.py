import logging

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.models.training.early_stopping import EarlyStopping
from pyclad.models.training.runners.standard import StandardRunner


class TinyBackbone(TorchBackbone):
    """Minimal backbone: base loss pushes the linear layer toward zero output."""

    def __init__(self):
        self.module = nn.Linear(2, 2)

    def get_module(self) -> nn.Module:
        return self.module

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.module.parameters(), lr=0.1)

    def compute_loss(self, x: Tensor) -> Tensor:
        return self.module(x).pow(2).mean()

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def name(self) -> str:
        return "tiny"


def _loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    data = torch.randn(n, 2)
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)


def _loss_fn(backbone: TinyBackbone):
    return lambda batch: backbone.compute_loss(batch[0])


def test_runs_all_epochs_without_early_stopping():
    backbone = TinyBackbone()
    result = StandardRunner(max_epochs=5).run(backbone, _loader(), _loss_fn(backbone))
    assert result.epochs_run == 5
    assert result.stopped_early is False
    assert result.best_val_loss is None
    assert result.final_val_loss is None


def test_validation_without_early_stopping_monitors_and_reports():
    # A validation set is meaningful on its own: it is evaluated every epoch and reported,
    # but nothing stops training early.
    backbone = TinyBackbone()
    runner = StandardRunner(max_epochs=3, validation_fraction=0.2)
    result = runner.run(
        backbone,
        _loader(),
        _loss_fn(backbone),
        val_loader=_loader(n=4),
        val_loss_fn=lambda _batch: 0.42,
    )
    assert result.epochs_run == 3
    assert result.stopped_early is False
    assert result.best_val_loss is None  # no monitor -> no "best"
    assert result.final_val_loss == 0.42  # monitoring still reports


@pytest.mark.parametrize("start_training", [True, False])
def test_run_preserves_module_training_mode(start_training):
    backbone = TinyBackbone()
    backbone.get_module().train(start_training)
    StandardRunner(max_epochs=2).run(backbone, _loader(), _loss_fn(backbone))
    assert backbone.get_module().training is start_training


@pytest.mark.parametrize("start_training", [True, False])
def test_run_with_early_stopping_preserves_module_training_mode(start_training):
    # The validation pass flips the module to eval(); it must not leak past run().
    backbone = TinyBackbone()
    backbone.get_module().train(start_training)
    runner = StandardRunner(max_epochs=3, validation_fraction=0.2, early_stopping=EarlyStopping(patience=5))
    runner.run(
        backbone,
        _loader(),
        _loss_fn(backbone),
        val_loader=_loader(n=4),
        val_loss_fn=lambda _batch: 0.5,
    )
    assert backbone.get_module().training is start_training


def test_warns_when_early_stopping_configured_but_no_validation(caplog):
    backbone = TinyBackbone()
    runner = StandardRunner(max_epochs=2, validation_fraction=0.2, early_stopping=EarlyStopping())
    with caplog.at_level(logging.WARNING):
        result = runner.run(backbone, _loader(), _loss_fn(backbone))
    assert result.epochs_run == 2
    assert result.stopped_early is False
    assert any("Early stopping is configured" in record.message for record in caplog.records)


def test_plain_runner_reports_itself():
    assert StandardRunner(max_epochs=3).info() == {
        "name": "StandardRunner",
        "max_epochs": 3,
        "validation_fraction": 0.0,
        "early_stopping": None,
    }


def test_reports_injected_early_stopping():
    runner = StandardRunner(max_epochs=3, validation_fraction=0.2, early_stopping=EarlyStopping(patience=4))
    info = runner.info()
    assert info["name"] == "StandardRunner"
    assert info["validation_fraction"] == 0.2
    # the monitor is reported as a serializable dict, not the object itself
    assert info["early_stopping"] == {"patience": 4, "min_delta": 0.0, "restore_best_weights": True}


def test_falls_back_to_fixed_length_without_validation_loader():
    backbone = TinyBackbone()
    runner = StandardRunner(max_epochs=4, validation_fraction=0.2, early_stopping=EarlyStopping(patience=1))
    # A monitor is configured, but no val_loader is passed -> plain fixed-length run.
    result = runner.run(backbone, _loader(), _loss_fn(backbone))
    assert result.epochs_run == 4
    assert result.stopped_early is False


def test_stops_when_validation_stalls():
    backbone = TinyBackbone()
    scripted = iter([1.0, 0.5, 0.6, 0.7, 0.8, 0.9])  # improves once, then degrades

    def val_loss_fn(_batch):
        return next(scripted)

    runner = StandardRunner(max_epochs=10, validation_fraction=0.2, early_stopping=EarlyStopping(patience=2))
    result = runner.run(
        backbone,
        _loader(),
        _loss_fn(backbone),
        val_loader=_loader(n=4),
        val_loss_fn=val_loss_fn,
    )

    assert result.stopped_early is True
    assert result.epochs_run < 10
    assert result.best_val_loss == 0.5  # best seen across epochs
    assert result.final_val_loss == 0.8  # last epoch before the stop triggered


def test_split_reflects_validation_fraction():
    data = np.arange(100).reshape(100, 1)
    assert StandardRunner(max_epochs=1).split_train_test(data)[1] is None  # no validation configured
    runner = StandardRunner(max_epochs=1, validation_fraction=0.25, early_stopping=EarlyStopping(), seed=0)
    train, val = runner.split_train_test(data)
    assert len(val) == 25 and len(train) == 75


def test_validation_without_early_stopping_is_allowed():
    # Validation is meaningful on its own (monitoring), so this must not raise.
    runner = StandardRunner(max_epochs=1, validation_fraction=0.2)
    assert runner.validation_fraction == 0.2
    assert runner.additional_info() == {"max_epochs": 1, "validation_fraction": 0.2, "early_stopping": None}


def test_early_stopping_requires_validation_fraction():
    with pytest.raises(ValueError):
        StandardRunner(max_epochs=1, early_stopping=EarlyStopping())
