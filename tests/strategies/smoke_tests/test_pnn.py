from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from pyclad.models.torch_backbone import TorchBackbone
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
        return self.forward(x).sum() * 0.0

    def forward(self, x: Tensor) -> Tensor:
        return self._module(x)

    def fit_with_loss(self, dataloader, loss_fn, epochs, grad_callback=None) -> None:
        pass

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._predict_calls == 0:
            scores = self._train_scores
        else:
            scores = self._predict_scores
        self._predict_calls += 1
        return np.full(scores.shape, self._label), scores

    def name(self) -> str:
        return "StaticScoreBackbone"


class NoOpLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.weight * 0.0


class EncoderDecoderModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = NoOpLayer()
        self.decoder = NoOpLayer()

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


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
        return np.zeros(scores.shape, dtype=int), scores

    def name(self) -> str:
        return "IncompatibleBackbone"


class CountingLinear(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.forward_calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.forward_calls += 1
        return self.linear(x)


class CountingBackbone(TorchBackbone):
    def __init__(self) -> None:
        self._module = nn.Module()
        self._module.encoder = CountingLinear()
        self._module.decoder = nn.Linear(1, 1)
        self.threshold = 0.5

    def get_module(self) -> nn.Module:
        return self._module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.SGD(self._module.parameters(), lr=0.01)

    def compute_loss(self, x: Tensor) -> Tensor:
        return ((self.forward(x) - x) ** 2).mean()

    def forward(self, x: Tensor) -> Tensor:
        return self._module.decoder(self._module.encoder(x))

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            scores = self.forward(torch.tensor(data, dtype=torch.float32)).detach().numpy().reshape(-1)
        return np.zeros(scores.shape, dtype=int), scores

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
        labels, scores = strategy.predict(data, concept_id="concept-a")

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

    def test_pnn_returns_zero_predictions_for_unknown_concepts_and_rejects_duplicates(self, backbone):
        data = np.random.default_rng(13).normal(0.0, 0.2, (24, self.FEATURE_DIM)).astype(np.float32)
        strategy = _strategy(backbone, task_free=False)

        strategy.learn(data, concept_id="concept-a")

        labels, scores = strategy.predict(data, concept_id="concept-b")

        assert labels.tolist() == [0.0] * len(data)
        assert scores.tolist() == [0.0] * len(data)
        with pytest.raises(ValueError, match="already learned"):
            strategy.learn(data, concept_id="concept-a")

    def test_task_free_pnn_selects_column_by_normalized_score(self):
        first_model = StaticScoreBackbone(train_scores=[0.9, 1.0, 1.1], predict_scores=[1.2], label=0)
        second_model = StaticScoreBackbone(train_scores=[], predict_scores=[], label=1)
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
        labels, scores = strategy.predict(np.zeros((1, 1), dtype=np.float32))

        assert labels.tolist() == [1]
        assert scores.shape == (1,)
        assert np.isfinite(scores).all()

    def test_task_free_pnn_uses_task_free_routing_even_with_concept_id(self):
        first_model = StaticScoreBackbone(train_scores=[0.9, 1.0, 1.1], predict_scores=[1.2], label=0)
        second_model = StaticScoreBackbone(train_scores=[], predict_scores=[], label=1)
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
        labels, _ = strategy.predict(np.zeros((1, 1), dtype=np.float32), concept_id="concept-a")

        assert labels.tolist() == [1]

    def test_pnn_rejects_backbones_without_encoder_decoder(self):
        strategy = PNNStrategy(base_model_factory=IncompatibleBackbone, task_free=True, epochs=1)

        with pytest.raises(ValueError, match="encoder and decoder"):
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
