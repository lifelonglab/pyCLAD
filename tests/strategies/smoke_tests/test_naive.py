import pytest

from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.strategies.baselines.naive import NaiveStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestNaiveStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return NaiveStrategy(TorchModelAdapter(backbone, epochs=2, batch_size=16))