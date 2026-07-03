import pytest

from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestCumulativeStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return CumulativeStrategy(TorchModelAdapter(backbone, StandardRunner(max_epochs=2), batch_size=16))
