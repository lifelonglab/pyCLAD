from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from pyclad.models.autoencoder.autoencoder import AutoencoderModule
from pyclad.models.torch_backbone import TorchBackbone
from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.architectural.pnn import PNNStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class StaticScoreBackbone(TorchBackbone):
    def __init__(self, train_scores: list[float], predict_scores: list[float], label: int) -> None:
        self._module = EncoderDecoderModule()
        self._train_scores = np.asarray(train_scores, dtype=np.float32)
        self._predict_scores = np.asarray(predict_scores, dtype=np.float32)
        self._label = label
        self._predict_calls = 0
        self.threshold = 0.5

    def get_module(self) -> nn.Module:
        return self._module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.SGD(self._module.parameters(), lr=0.01)

    def compute_loss(self, x: Tensor) -> Tensor:
        return (self.forward(x) - x).abs().mean()

    def forward(self, x: Tensor) -> Tensor:
        return self._module(x)

    def fit_with_loss(self, dataloader, loss_fn, epochs, grad_callback=None) -> None:
        pass

    def predict(self, data):
        self.forward(torch.tensor(data, dtype=torch.float32))
        scores = self._train_scores if self._predict_calls == 0 else self._predict_scores
        self._predict_calls += 1
        return PredictionResults(y_pred=np.full(len(scores), self._label), anomaly_scores=scores)

    def name(self) -> str:
        return "StaticScoreBackbone"


class EncoderDecoderModule(AutoencoderModule):
    def __init__(self) -> None:
        super().__init__(nn.Linear(1, 1), nn.Linear(1, 1))


class IncompatibleBackbone(TorchBackbone):
    def __init__(self) -> None:
        self._module = nn.Linear(1, 1)

    def get_module(self) -> nn.Module:
        return self._module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.SGD(self._module.parameters(), lr=0.01)

    def compute_loss(self, x: Tensor) -> Tensor:
        return self.forward(x).sum() * 0.0

    def forward(self, x: Tensor) -> Tensor:
        return self._module(x)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(data.shape[0], dtype=np.float32)
        return PredictionResults(y_pred=np.zeros(scores.shape, dtype=int), anomaly_scores=scores)

    def name(self) -> str:
        return "IncompatibleBackbone"


class CountingLinear(nn.Linear):
    def __init__(self, in_features: int = 1, out_features: int = 1) -> None:
        super().__init__(in_features, out_features)
        self.forward_calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.forward_calls += 1
        return super().forward(x)


class CountingBackbone(TorchBackbone):
    def __init__(self) -> None:
        self._module = AutoencoderModule(CountingLinear(), nn.Linear(1, 1))
        self.threshold = 0.5

    def get_module(self) -> nn.Module:
        return self._module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.SGD(self._module.parameters(), lr=0.01)

    def compute_loss(self, x: Tensor) -> Tensor:
        return ((self.forward(x) - x) ** 2).mean()

    def forward(self, x: Tensor) -> Tensor:
        return self._module(x)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            scores = self.forward(torch.tensor(data, dtype=torch.float32)).detach().numpy().reshape(-1)
        return PredictionResults(y_pred=np.zeros(scores.shape, dtype=int), anomaly_scores=scores)

    def name(self) -> str:
        return "CountingBackbone"


def _strategy(backbone, task_free: bool) -> PNNStrategy:
    return PNNStrategy(
        base_model_factory=lambda: deepcopy(backbone),
        batch_size=16,
        epochs=2,
        task_free=task_free,
        device="cpu",
    )


class TestPNNStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return _strategy(backbone, task_free=True)

    def test_concept_aware_pnn_predicts_by_concept_id(self, backbone):
        data = np.random.default_rng(7).normal(0.0, 0.2, (24, self.FEATURE_DIM)).astype(np.float32)
        strategy = _strategy(backbone, task_free=False)

        strategy.learn(data, concept_id="concept-a")
        result = strategy.predict(data, concept_id="concept-a")
        labels, scores = result.y_pred, result.anomaly_scores

        assert labels.shape == (len(data),)
        assert scores.shape == (len(data),)

    def test_concept_aware_pnn_requires_concept_id(self, backbone):
        data = np.random.default_rng(11).normal(0.0, 0.2, (24, self.FEATURE_DIM)).astype(np.float32)
        strategy = _strategy(backbone, task_free=False)

        with pytest.raises(ValueError, match="requires concept_id"):
            strategy.learn(data)

        strategy.learn(data, concept_id="concept-a")

        with pytest.raises(ValueError, match="requires concept_id"):
            strategy.predict(data)

    def test_pnn_rejects_unknown_concepts_and_duplicates(self, backbone):
        data = np.random.default_rng(13).normal(0.0, 0.2, (24, self.FEATURE_DIM)).astype(np.float32)
        strategy = _strategy(backbone, task_free=False)

        strategy.learn(data, concept_id="concept-a")

        with pytest.raises(ValueError, match="Unknown concept_id"):
            strategy.predict(data, concept_id="concept-b")
        with pytest.raises(ValueError, match="already learned"):
            strategy.learn(data, concept_id="concept-a")

    def test_task_free_pnn_selects_column_by_normalized_score(self):
        first_model = StaticScoreBackbone(train_scores=[0.9, 1.0, 1.1], predict_scores=[1.2], label=0)
        second_model = StaticScoreBackbone(train_scores=[9.0, 10.0, 11.0], predict_scores=[10.0], label=1)
        second_model.threshold = -1.0
        models = iter(
            [
                first_model,
                second_model,
            ]
        )
        strategy = PNNStrategy(base_model_factory=lambda: next(models), task_free=True, epochs=1)

        strategy.learn(np.zeros((3, 1), dtype=np.float32))
        strategy.learn(np.ones((3, 1), dtype=np.float32))
        result = strategy.predict(np.zeros((1, 1), dtype=np.float32))
        labels, scores = result.y_pred, result.anomaly_scores

        assert labels.tolist() == [1]
        assert scores.shape == (1,)
        assert np.isfinite(scores).all()

    def test_task_free_pnn_uses_task_free_routing_even_with_concept_id(self):
        first_model = StaticScoreBackbone(train_scores=[0.9, 1.0, 1.1], predict_scores=[1.2], label=0)
        second_model = StaticScoreBackbone(train_scores=[9.0, 10.0, 11.0], predict_scores=[10.0], label=1)
        second_model.threshold = -1.0
        models = iter(
            [
                first_model,
                second_model,
            ]
        )
        strategy = PNNStrategy(base_model_factory=lambda: next(models), task_free=True, epochs=1)

        strategy.learn(np.zeros((3, 1), dtype=np.float32), concept_id="concept-a")
        strategy.learn(np.ones((3, 1), dtype=np.float32), concept_id="concept-b")
        result = strategy.predict(np.zeros((1, 1), dtype=np.float32), concept_id="concept-a")
        labels, _ = result.y_pred, result.anomaly_scores

        assert labels.tolist() == [1]

    def test_pnn_rejects_unsupported_architectures(self):
        strategy = PNNStrategy(base_model_factory=IncompatibleBackbone, task_free=True, epochs=1)

        with pytest.raises(ValueError, match="AutoencoderModule"):
            strategy.learn(np.zeros((3, 1), dtype=np.float32))

    def test_later_columns_receive_previous_column_activations(self):
        first_model = CountingBackbone()
        second_model = CountingBackbone()
        models = iter([first_model, second_model])
        strategy = PNNStrategy(base_model_factory=lambda: next(models), task_free=True, epochs=1, batch_size=4)

        strategy.learn(np.zeros((4, 1), dtype=np.float32))
        calls_after_first_column = first_model.get_module().encoder.forward_calls
        strategy.learn(np.ones((4, 1), dtype=np.float32))

        assert first_model.get_module().encoder.forward_calls > calls_after_first_column


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_calibration_rolls_back_and_allows_retry(bad_score):
    models = iter(
        [
            StaticScoreBackbone([bad_score] * 3, [0.0], 0),
            StaticScoreBackbone([1.0] * 3, [1.0], 0),
        ]
    )
    strategy = PNNStrategy(lambda: next(models), epochs=1)
    data = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="finite anomaly score"):
        strategy.learn(data, concept_id="a")
    assert strategy.num_columns == 0
    assert strategy._score_stats == []
    assert strategy._concept_to_column == {}
    strategy.learn(data, concept_id="a")
    assert strategy._score_stats == [(1.0, 1.0)]


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf")])
def test_task_free_rejects_nonfinite_prediction(bad_score):
    strategy = PNNStrategy(lambda: StaticScoreBackbone([1.0] * 3, [bad_score], 0), epochs=1, task_free=True)
    strategy.learn(np.zeros((3, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="finite anomaly score"):
        strategy.predict(np.zeros((1, 1), dtype=np.float32))


def test_custom_loss_uses_original_input_and_progressive_output_for_all_columns():
    class RecordingBackbone(StaticScoreBackbone):
        def get_optimizer(self):
            return torch.optim.SGD(self._module.parameters(), lr=0.0)

        def compute_loss(self, x):
            output = self.forward(x)
            self.loss_input, self.loss_output = x.detach().clone(), output.detach().clone()
            return (output - x).abs().mean()

    strategy = PNNStrategy(lambda: RecordingBackbone([1.0] * 3, [1.0] * 3, 0), epochs=1, task_free=True)
    data = np.ones((3, 1), dtype=np.float32)
    strategy.learn(data)
    strategy.learn(data)
    for index, column in enumerate(strategy._columns):
        torch.testing.assert_close(column.loss_input, torch.tensor(data))
        torch.testing.assert_close(column.loss_output, strategy._forward(index, torch.tensor(data)))


def test_adapters_update_and_previous_columns_remain_frozen():
    torch.manual_seed(7)
    strategy = PNNStrategy(CountingBackbone, epochs=2, task_free=True)
    data = np.ones((4, 1), dtype=np.float32)
    strategy.learn(data)
    old = [p.detach().clone() for p in strategy._params(0)]
    original_fit = strategy._fit
    updates = []

    def checked_fit(index, batch):
        before = [p.detach().clone() for p in strategy._adapters[index].parameters()]
        original_fit(index, batch)
        updates.extend(not torch.equal(a, b) for a, b in zip(before, strategy._adapters[index].parameters()))

    strategy._fit = checked_fit
    strategy.learn(data)
    assert updates and any(updates)
    assert all(torch.equal(a, b) for a, b in zip(old, strategy._params(0)))
    assert all(not p.requires_grad for index in range(2) for p in strategy._params(index))


def test_unsupported_module_is_rejected_without_adding_column():
    class BlocksBackbone(CountingBackbone):
        def __init__(self):
            self._module = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))

    strategy = PNNStrategy(BlocksBackbone, epochs=1, task_free=True)
    with pytest.raises(ValueError, match="AutoencoderModule"):
        strategy.learn(np.ones((4, 1), dtype=np.float32))
    assert strategy.num_columns == 0


def test_training_failure_preserves_existing_column():
    strategy = PNNStrategy(CountingBackbone, epochs=1)
    data = np.ones((4, 1), dtype=np.float32)
    strategy.learn(data, concept_id="a")
    before = strategy.predict(data, concept_id="a").anomaly_scores.copy()

    def fail_fit(index, batch):
        raise RuntimeError("training failed")

    strategy._fit = fail_fit
    with pytest.raises(RuntimeError, match="training failed"):
        strategy.learn(data, concept_id="b")
    assert strategy.num_columns == 1
    assert strategy._concept_to_column == {"a": 0}
    assert len(strategy._score_stats) == 1
    np.testing.assert_array_equal(before, strategy.predict(data, concept_id="a").anomaly_scores)


def test_adapter_shape_probe_does_not_update_batch_norm_statistics():
    class BatchNormBackbone(CountingBackbone):
        def __init__(self):
            self._module = AutoencoderModule(nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2)), nn.Linear(2, 1))

    strategy = PNNStrategy(BatchNormBackbone, epochs=1, batch_size=4, task_free=True)
    data = np.arange(4, dtype=np.float32).reshape(-1, 1)
    strategy.learn(data)
    strategy.learn(data)
    assert all(column.get_module().encoder[1].num_batches_tracked.item() == 1 for column in strategy._columns)


def test_optimizer_preserves_backbone_parameter_groups(monkeypatch):
    optimizers = []
    original_step = torch.optim.SGD.step

    def capture_step(optimizer, *args, **kwargs):
        optimizers.append(optimizer)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", capture_step)

    class GroupedBackbone(CountingBackbone):
        def get_optimizer(self):
            return torch.optim.SGD(
                [
                    {"params": self._module.encoder.parameters(), "lr": 0.03},
                    {"params": self._module.decoder.parameters(), "lr": 0.07},
                ],
                lr=0.01,
            )

    strategy = PNNStrategy(GroupedBackbone, epochs=1, task_free=True)
    strategy.learn(np.ones((4, 1), dtype=np.float32))
    strategy.learn(np.ones((4, 1), dtype=np.float32))
    optimizer = optimizers[-1]
    assert [group["lr"] for group in optimizer.param_groups] == [0.03, 0.07]
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in strategy._params(1)}


@pytest.mark.parametrize("shared", ["model", "layer"])
def test_factory_cannot_reuse_previous_column_state(shared):
    first = CountingBackbone()
    second = first if shared == "model" else CountingBackbone()
    second._module.encoder = first._module.encoder
    models = iter([first, second])
    strategy = PNNStrategy(lambda: next(models), epochs=1)
    data = np.ones((4, 1), dtype=np.float32)
    strategy.learn(data, concept_id="a")
    before = strategy.predict(data, concept_id="a").anomaly_scores.copy()
    with pytest.raises(ValueError, match="independent"):
        strategy.learn(data, concept_id="b")
    assert strategy.num_columns == 1
    np.testing.assert_array_equal(before, strategy.predict(data, concept_id="a").anomaly_scores)


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_training_cannot_hide_behind_finite_scores(failure):
    class BadTrainingBackbone(StaticScoreBackbone):
        def compute_loss(self, x):
            output = self.forward(x)
            if failure == "loss":
                return output.mean() * float("nan")
            output.register_hook(lambda grad: torch.full_like(grad, float("inf")))
            return output.mean()

    strategy = PNNStrategy(lambda: BadTrainingBackbone([1.0] * 3, [1.0] * 3, 0), epochs=1)
    with pytest.raises(ValueError, match="finite"):
        strategy.learn(np.ones((3, 1), dtype=np.float32), concept_id="a")
    assert strategy.num_columns == 0
    assert not strategy._concept_to_column


def test_each_previous_layer_runs_once_per_forward():
    strategy = PNNStrategy(CountingBackbone, epochs=1, task_free=True)
    data = np.ones((4, 1), dtype=np.float32)
    for _ in range(3):
        strategy.learn(data)
    for column in strategy._columns:
        column.get_module().encoder.forward_calls = 0
    strategy._forward(2, torch.tensor(data))
    assert [column.get_module().encoder.forward_calls for column in strategy._columns] == [1, 1, 1]


@pytest.mark.parametrize("data", [np.ones(3), np.empty((0, 1)), np.array([[np.nan]])])
def test_prediction_rejects_invalid_data_even_with_constant_scores(data):
    strategy = PNNStrategy(lambda: StaticScoreBackbone([1.0] * 3, [1.0] * len(data), 0), epochs=1, task_free=True)
    strategy.learn(np.ones((3, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="finite two-dimensional"):
        strategy.predict(data)


@pytest.mark.parametrize("adapter_kind", ["linear", "mlp"])
@pytest.mark.parametrize("inplace", [False, True])
def test_forward_matches_original_pnn_equations_and_gradients(adapter_kind, inplace):
    from pyclad.models.autoencoder.autoencoder import Autoencoder

    torch.manual_seed(17)
    widths = iter([3, 4, 5])

    def factory():
        width = next(widths)
        return Autoencoder(
            nn.Sequential(nn.Linear(2, width), nn.ReLU(inplace=inplace)),
            nn.Sequential(nn.Linear(width, 2), nn.Sigmoid()),
        )

    strategy = PNNStrategy(
        factory,
        epochs=1,
        task_free=True,
        adapter=adapter_kind,
    )
    data = np.random.default_rng(8).normal(size=(4, 2)).astype(np.float32)
    for _ in range(3):
        strategy.learn(data)
    strategy._set_trainable(2, True)
    x = torch.tensor(data)

    # Independent, depth-first evaluation of Eqs. 1 and 2. Each lateral consumes
    # the preceding depth; the outer activation follows the combined preactivation.
    previous = [x, x, x]
    for depth in range(2):
        current = []
        for index, column in enumerate(strategy._columns):
            block = column.get_module().encoder if depth == 0 else column.get_module().decoder
            value = block[0](previous[index])
            if index:
                adapter = strategy._adapters[index][depth]
                inputs = [old.detach() for old in previous[:index]]
                if adapter_kind == "linear":
                    for projection, old in zip(adapter.projections, inputs):
                        value = value + torch.nn.functional.linear(old, projection.weight, projection.bias)
                else:
                    scaled = torch.cat([adapter.scales[j] * old for j, old in enumerate(inputs)], dim=1)
                    v, _, u = adapter.projection
                    lateral = torch.nn.functional.linear(scaled, v.weight, v.bias).relu()
                    value = value + torch.nn.functional.linear(lateral, u.weight, u.bias)
            value = value.relu() if depth == 0 else value.sigmoid()
            current.append(value.detach() if index < 2 else value)
        previous = current

    actual = strategy._forward(2, x)
    expected = previous[2]
    torch.testing.assert_close(actual, expected)
    params = list(strategy._params(2))
    actual_grads = torch.autograd.grad(actual.sum(), params)
    expected_grads = torch.autograd.grad(expected.sum(), params)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert all(not p.requires_grad for index in range(2) for p in strategy._params(index))


@pytest.mark.parametrize("adapter_kind", ["linear", "mlp"])
def test_laterals_cannot_bypass_output_sigmoid(adapter_kind):
    from pyclad.models.autoencoder.autoencoder import Autoencoder

    strategy = PNNStrategy(
        lambda: Autoencoder(nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())),
        epochs=1,
        task_free=True,
        adapter=adapter_kind,
    )
    x = torch.ones(4, 2)
    strategy.learn(x.numpy())
    strategy.learn(x.numpy())
    with torch.no_grad():
        for parameter in strategy._adapters[1].parameters():
            parameter.fill_(-10.0)
        output = strategy._forward(1, x)
    assert torch.isfinite(output).all()
    assert ((output >= 0) & (output <= 1)).all()
    assert len(strategy._adapters[1]) == 2  # One adapter per transform, none after sigmoid.


def test_autoencoder_rejects_opaque_blocks_for_pnn_only():
    from pyclad.models.autoencoder.autoencoder import Autoencoder

    class OpaqueBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x).sigmoid()

    model = Autoencoder(OpaqueBlock(), nn.Linear(2, 2))
    assert model.forward(torch.ones(4, 2)).shape == (4, 2)
    strategy = PNNStrategy(lambda: model, epochs=1)
    with pytest.raises(ValueError, match="explicit dense"):
        strategy.learn(np.ones((4, 2), dtype=np.float32), concept_id="a")
    assert strategy.num_columns == 0


def test_first_column_matches_nested_autoencoder():
    from pyclad.models.autoencoder.autoencoder import Autoencoder

    strategy = PNNStrategy(
        lambda: Autoencoder(
            nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.ReLU())), nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())
        ),
        epochs=1,
        task_free=True,
    )
    x = torch.ones(4, 2)
    strategy.learn(x.numpy())
    torch.testing.assert_close(strategy._forward(0, x), strategy._columns[0].forward(x))
    assert not list(strategy._adapters[0].parameters())


def test_nonfinite_optimizer_update_rolls_back(monkeypatch):
    def overflow(optimizer, *args, **kwargs):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("inf"))

    monkeypatch.setattr(torch.optim.SGD, "step", overflow)
    strategy = PNNStrategy(lambda: StaticScoreBackbone([1.0] * 3, [1.0] * 3, 0), epochs=1)
    with pytest.raises(ValueError, match="finite parameters"):
        strategy.learn(np.ones((3, 1), dtype=np.float32), concept_id="a")
    assert strategy.num_columns == 0
    assert not strategy._score_stats
    assert not strategy._concept_to_column


@pytest.mark.parametrize("adapter_kind", ["linear", "mlp"])
def test_learning_third_column_preserves_all_previous_predictions(adapter_kind):
    strategy = PNNStrategy(CountingBackbone, epochs=2, adapter=adapter_kind)
    data = np.arange(4, dtype=np.float32).reshape(-1, 1)
    strategy.learn(data, concept_id="a")
    strategy.learn(data + 1, concept_id="b")
    before = {concept: strategy.predict(data, concept_id=concept).anomaly_scores.copy() for concept in ("a", "b")}
    strategy.learn(data - 1, concept_id="c")
    for concept, scores in before.items():
        np.testing.assert_array_equal(scores, strategy.predict(data, concept_id=concept).anomaly_scores)


def test_invalid_adapter_is_rejected():
    with pytest.raises(ValueError, match="adapter must"):
        PNNStrategy(CountingBackbone, adapter="unknown")


def test_runner_validation_and_prediction_use_progressive_outputs():
    from pyclad.models.autoencoder.autoencoder import Autoencoder
    from pyclad.models.training.runners.standard import StandardRunner

    runner = StandardRunner(max_epochs=2, validation_fraction=0.25, seed=7)
    strategy = PNNStrategy(
        lambda: Autoencoder(nn.Sequential(nn.Linear(2, 3), nn.ReLU()), nn.Linear(3, 2)),
        runner=runner,
        batch_size=4,
    )
    data = np.random.default_rng(7).normal(size=(16, 2)).astype(np.float32)
    strategy.learn(data, concept_id="a")
    strategy.learn(data, concept_id="b")
    result = strategy.predict(data, concept_id="b")
    with torch.no_grad():
        output = strategy._forward(1, torch.tensor(data)).numpy()
    scores = ((data - output) ** 2).mean(axis=1)
    np.testing.assert_allclose(result.anomaly_scores, scores, rtol=1e-5)
    np.testing.assert_array_equal(result.y_pred, scores > strategy._columns[1].threshold)
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for column in strategy._columns
        for module in column.get_module().modules()
    )


def test_lateral_hooks_are_removed_when_model_prediction_raises():
    strategy = PNNStrategy(CountingBackbone, epochs=1)
    data = np.ones((4, 1), dtype=np.float32)
    strategy.learn(data, concept_id="a")
    strategy.learn(data, concept_id="b")

    def fail(data):
        raise RuntimeError("prediction failed")

    strategy._columns[1].predict = fail
    with pytest.raises(RuntimeError, match="prediction failed"):
        strategy.predict(data, concept_id="b")
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in strategy._columns[1].get_module().modules()
    )


@pytest.mark.parametrize("restore", [True, False])
def test_early_stopping_starts_fresh_for_each_column(restore):
    from pyclad.models.autoencoder.autoencoder import Autoencoder
    from pyclad.models.training.early_stopping import EarlyStopping
    from pyclad.models.training.runners.standard import StandardRunner

    def factory():
        model = Autoencoder(nn.Linear(2, 2), nn.Linear(2, 2), lr=0.0)
        with torch.no_grad():
            for parameter in model.get_module().parameters():
                parameter.zero_()
        return model

    results = []

    class RecordingRunner(StandardRunner):
        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            results.append(result)
            return result

    stopper = EarlyStopping(patience=1, restore_best_weights=restore)
    runner = RecordingRunner(max_epochs=5, validation_fraction=0.25, early_stopping=stopper, seed=7)
    strategy = PNNStrategy(factory, runner=runner)
    data = np.zeros((8, 2), dtype=np.float32)
    strategy.learn(data, concept_id="a")
    assert results[0].best_val_loss == 0.0
    before = strategy.predict(data, concept_id="a").anomaly_scores.copy()
    strategy.learn(data + 100, concept_id="b")
    assert strategy.num_columns == 2
    assert results[1].best_val_loss > 0.0
    assert [result.epochs_run for result in results] == [3, 3]
    np.testing.assert_array_equal(before, strategy.predict(data, concept_id="a").anomaly_scores)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_learning_and_prediction_on_requested_device(device):
    from pyclad.models.autoencoder.autoencoder import Autoencoder

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    strategy = PNNStrategy(lambda: Autoencoder(nn.Linear(2, 3), nn.Linear(3, 2)), device=device, epochs=1)
    data = np.ones((8, 2), dtype=np.float32)
    for concept in ("a", "b"):
        strategy.learn(data, concept_id=concept)
        result = strategy.predict(data, concept_id=concept)
        with torch.no_grad():
            output = strategy._forward(strategy.current_column, torch.tensor(data, device=device)).cpu().numpy()
        np.testing.assert_allclose(result.anomaly_scores, ((data - output) ** 2).mean(axis=1), rtol=1e-5)
        np.testing.assert_array_equal(result.y_pred, result.anomaly_scores > 0.5)


def test_reused_dense_transform_is_rejected_before_learning():
    from pyclad.models.autoencoder.autoencoder import Autoencoder

    shared = nn.Linear(2, 2)
    model = Autoencoder(nn.Sequential(shared, nn.ReLU()), shared)
    before = [parameter.detach().clone() for parameter in model.get_module().parameters()]
    strategy = PNNStrategy(lambda: model, epochs=1)
    with pytest.raises(ValueError, match="distinct"):
        strategy.learn(np.ones((4, 2), dtype=np.float32), concept_id="a")
    assert strategy.num_columns == 0
    for old, parameter in zip(before, model.get_module().parameters()):
        torch.testing.assert_close(old, parameter)
