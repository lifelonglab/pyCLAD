import pytest

from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.replay.agem import AGEMStrategy
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestAGEMStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        buffer = ReservoirBuffer(max_capacity=100, device="cpu")
        return AGEMStrategy(
            model=backbone,
            runner=StandardRunner(max_epochs=2),
            buffer=buffer,
            batch_size=16,
            replay_batch_size=16,
            device="cpu",
        )
