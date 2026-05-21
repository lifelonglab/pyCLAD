from copy import deepcopy

import pytest

from pyclad.strategies.architectural.pnn import PNNStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestPNNStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return PNNStrategy(
            base_model_factory=lambda: deepcopy(backbone),
            batch_size=16,
            epochs=2,
            task_free=True,
            device="cpu",
        )
