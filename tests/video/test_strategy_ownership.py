"""Exercise COMMAND as a model and ContTrain++ as its training strategy."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.video.data.sample import VideoBag
from pyclad.video.models.command.command import CommandModel
from pyclad.video.models.command.config import CommandArchitectureConfig
from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy


def test_command_can_compute_loss_and_predict_without_a_continual_trainer():
    """A model consumer must be able to train a labelled batch without replay or task setup."""
    assert issubclass(CommandModel, TorchBackbone)
    from pyclad.video.models.command.config import CommandConfig

    architecture = CommandArchitectureConfig(
        appearance_dim=4, motion_dim=4, hidden_dim=8, state_dim=3, memory_size=5, cbam_reduction=2, dropout=0.0
    )
    model = CommandModel(config=CommandConfig(architecture=architecture, threshold=0.0))
    features = torch.randn(4, 8, 8)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    optimizer = model.get_optimizer()
    before = model.get_module().network.classifier.weight.detach().clone()
    optimizer.zero_grad()
    model.compute_loss(features, labels).backward()
    optimizer.step()
    assert not torch.equal(before, model.get_module().network.classifier.weight)
    bags = np.asarray([VideoBag(str(i), "test", x.numpy(), int(y)) for i, (x, y) in enumerate(zip(features, labels))])
    prediction = model.predict(bags)
    assert prediction.y_pred.shape == (4,)
    assert prediction.window_scores.shape == (32,)
    assert np.isfinite(prediction.window_scores).all()
    assert model.forward(features).shape == (4, 8)


def test_cont_train_owns_persistent_replay_and_checkpoint_continuation(tmp_path):
    """Training and resumable replay must be available on the strategy itself."""
    assert callable(getattr(ContTrainPlusPlusStrategy, "fit_task", None))
    from pyclad.video.models.command.config import CommandConfig
    from pyclad.video.strategies.config import ContTrainPlusPlusConfig

    architecture = CommandArchitectureConfig(
        appearance_dim=4, motion_dim=4, hidden_dim=8, state_dim=3, memory_size=5, cbam_reduction=2, dropout=0.25
    )
    model_config = CommandConfig(architecture=architecture, seed=17)
    strategy_config = ContTrainPlusPlusConfig(epochs=1, batch_size=4, replay_batch_size=2, buffer_size=6, seed=17)
    model = CommandModel(config=model_config)
    strategy = ContTrainPlusPlusStrategy(model, config=strategy_config)
    bags = tuple(VideoBag(f"T1-{i}", "T1", np.random.default_rng(i).normal(size=(8, 8)), i % 2) for i in range(4))
    strategy.learn(np.asarray(bags, dtype=object))
    checkpoint = tmp_path / "strategy.pt"
    strategy.save_checkpoint(checkpoint)
    next_bags = tuple(replace(bag, bag_id=bag.bag_id.replace("T1", "T2"), task_id="T2") for bag in bags)
    strategy.learn(np.asarray(next_bags, dtype=object))
    restored_model = CommandModel(config=model_config)
    restored = ContTrainPlusPlusStrategy(restored_model, config=strategy_config)
    restored.load_checkpoint(checkpoint)
    restored.learn(np.asarray(next_bags, dtype=object))
    assert {entry.bag.task_id for entry in restored.replay.entries} == {"T1", "T2"}
    for key, value in model.get_module().state_dict().items():
        torch.testing.assert_close(value, restored_model.get_module().state_dict()[key], rtol=0, atol=0)
    assert strategy.info()["strategy"]["global_epochs"] == 2
    assert strategy.info()["strategy"]["buffer_bags"] == 6
    assert "training" not in model.info()["model"]


@pytest.mark.parametrize("operation", ["calibrate", "predict_bags", "fit_task"])
def test_model_and_strategy_accept_core_object_array_rows(operation):
    """Direct video APIs must accept the same object-array rows as a core concept."""
    from pyclad.video.models.command.config import CommandConfig
    from pyclad.video.strategies.config import ContTrainPlusPlusConfig

    architecture = CommandArchitectureConfig(
        appearance_dim=4, motion_dim=4, hidden_dim=8, state_dim=3, memory_size=5, cbam_reduction=2, dropout=0.0
    )
    model = CommandModel(config=CommandConfig(architecture=architecture))
    bags = np.asarray(
        [VideoBag(str(i), "T1", np.random.default_rng(i).normal(size=(8, 8)), 0) for i in range(4)], dtype=object
    )
    if operation == "calibrate":
        model.calibrate(bags)
        assert model.predict(bags).y_pred.shape == (4,)
    elif operation == "predict_bags":
        assert model.predict_bags(bags)["raw_scores"].shape == (4, 8)
    else:
        strategy = ContTrainPlusPlusStrategy(
            model, config=ContTrainPlusPlusConfig(epochs=1, batch_size=4, replay_batch_size=2)
        )
        result = strategy.fit_task(bags)
        assert result["bags"] == 4
        assert len(strategy.replay) == 4
