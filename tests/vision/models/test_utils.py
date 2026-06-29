import copy
import types

import pytest
import torch
from torch import nn

from pyclad.vision.models.utilities.utils import BestWeightsCallback


def _trainer(loss: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(callback_metrics={"train_loss": torch.tensor(loss)})


def test_best_weights_callback_tracks_min_delta_improvements():
    module = types.SimpleNamespace(network=nn.Linear(2, 2))
    callback = BestWeightsCallback(monitor="train_loss", min_delta=0.1)

    # First epoch: baseline recorded and current weights captured.
    callback.on_train_epoch_end(_trainer(1.0), module)
    assert callback.best_loss == pytest.approx(1.0)
    assert callback.best_state_dict is not None
    epoch0_weight = copy.deepcopy(module.network.state_dict()["weight"])
    assert torch.equal(callback.best_state_dict["weight"], epoch0_weight)

    # Mutate weights, then report a sub-min_delta improvement: best must NOT update.
    with torch.no_grad():
        module.network.weight.add_(1.0)
    callback.on_train_epoch_end(_trainer(0.95), module)  # delta 0.05 < 0.1
    assert callback.best_loss == pytest.approx(1.0)
    assert torch.equal(callback.best_state_dict["weight"], epoch0_weight)

    # A larger-than-min_delta improvement: best updates to the current (mutated) weights.
    callback.on_train_epoch_end(_trainer(0.8), module)  # delta 0.2 > 0.1
    assert callback.best_loss == pytest.approx(0.8)
    assert torch.equal(callback.best_state_dict["weight"], module.network.state_dict()["weight"])


def test_best_weights_callback_ignores_missing_monitor_key():
    module = types.SimpleNamespace(network=nn.Linear(2, 2))
    callback = BestWeightsCallback(monitor="train_loss", min_delta=0.0)

    callback.on_train_epoch_end(types.SimpleNamespace(callback_metrics={}), module)

    assert callback.best_loss is None
    assert callback.best_state_dict is None
