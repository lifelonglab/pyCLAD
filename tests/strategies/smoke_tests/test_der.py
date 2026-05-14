import pytest

from pyclad.strategies.regularization.der import DerPlusPlus
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestDerPlusPlus(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        buffer = ReservoirBuffer(max_capacity=100, device="cpu")
        return DerPlusPlus(model=backbone, buffer=buffer, alpha=0.5, beta=0.5, batch_size=16, epochs=2, device="cpu")