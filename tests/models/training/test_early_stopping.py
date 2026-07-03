import pytest
import torch
from torch import nn

from pyclad.models.training.early_stopping import EarlyStopping


def _module(value: float) -> nn.Module:
    module = nn.Linear(1, 1)
    with torch.no_grad():
        module.weight.fill_(value)
        module.bias.zero_()
    return module


def test_stops_after_patience_without_improvement():
    stopper = EarlyStopping(patience=2)
    module = _module(0.0)

    assert stopper.step(1.0, module) is False  # improvement (baseline)
    assert stopper.step(2.0, module) is False  # no improvement, 1 > patience? no
    assert stopper.step(2.0, module) is False  # 2nd bad epoch, still within patience
    assert stopper.step(2.0, module) is True  # 3rd bad epoch exceeds patience -> stop


def test_improvement_resets_the_counter():
    stopper = EarlyStopping(patience=1)
    module = _module(0.0)

    stopper.step(1.0, module)
    assert stopper.step(2.0, module) is False  # 1 bad epoch
    assert stopper.step(0.5, module) is False  # improvement resets
    assert stopper.step(2.0, module) is False  # 1 bad epoch again
    assert stopper.step(2.0, module) is True  # now exceeds patience


def test_min_delta_requires_meaningful_improvement():
    stopper = EarlyStopping(patience=0, min_delta=0.5)
    module = _module(0.0)

    stopper.step(1.0, module)
    # 0.6 is lower but not by min_delta, so it counts as no improvement and stops (patience=0)
    assert stopper.step(0.6, module) is True
    assert stopper.best_loss == 1.0


def test_restore_loads_best_weights():
    stopper = EarlyStopping(patience=5, restore_best_weights=True)
    module = _module(1.0)

    stopper.step(0.1, module)  # best snapshot taken at weight=1.0
    with torch.no_grad():
        module.weight.fill_(9.0)  # drift away from best
    stopper.step(0.5, module)  # worse, no new snapshot

    stopper.restore(module)
    assert module.weight.item() == pytest.approx(1.0)


def test_restore_is_noop_when_disabled():
    stopper = EarlyStopping(patience=5, restore_best_weights=False)
    module = _module(1.0)
    stopper.step(0.1, module)
    with torch.no_grad():
        module.weight.fill_(9.0)
    stopper.restore(module)
    assert module.weight.item() == pytest.approx(9.0)


@pytest.mark.parametrize("patience,min_delta", [(-1, 0.0), (0, -0.1)])
def test_invalid_arguments_rejected(patience, min_delta):
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience, min_delta=min_delta)
