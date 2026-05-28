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
        self._module = nn.Linear(1, 1)
        self._train_scores = np.asarray(train_scores, dtype=np.float32)
        self._predict_scores = np.asarray(predict_scores, dtype=np.float32)
        self._label = label
        self._predict_calls = 0

    def get_module(self) -> nn.Module:
        return self._module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.SGD(self._module.parameters(), lr=0.01)

    def compute_loss(self, x: Tensor) -> Tensor:
        return self._module(x).sum() * 0.0

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

    def test_pnn_rejects_unknown_and_duplicate_concepts(self, backbone):
        data = np.random.default_rng(13).normal(0.0, 0.2, (24, self.FEATURE_DIM)).astype(np.float32)
        strategy = _strategy(backbone, task_free=False)

        strategy.learn(data, concept_id="concept-a")

        with pytest.raises(ValueError, match="Unknown concept_id"):
            strategy.predict(data, concept_id="concept-b")
        with pytest.raises(ValueError, match="already learned"):
            strategy.learn(data, concept_id="concept-a")

    def test_task_free_pnn_selects_column_by_normalized_score(self):
        models = iter(
            [
                StaticScoreBackbone(train_scores=[0.9, 1.0, 1.1], predict_scores=[1.2], label=0),
                StaticScoreBackbone(train_scores=[90.0, 100.0, 110.0], predict_scores=[101.0], label=1),
            ]
        )
        strategy = PNNStrategy(base_model_factory=lambda: next(models), task_free=True, epochs=1)

        strategy.learn(np.zeros((3, 1), dtype=np.float32))
        strategy.learn(np.ones((3, 1), dtype=np.float32))
        labels, scores = strategy.predict(np.zeros((1, 1), dtype=np.float32))

        assert labels.tolist() == [1]
        assert scores[0] == pytest.approx((101.0 - 100.0) / np.std([90.0, 100.0, 110.0]))
