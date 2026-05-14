import pytest

from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestReplayEnhancedStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=200)
        return ReplayEnhancedStrategy(model=TorchModelAdapter(backbone, epochs=2, batch_size=16), buffer=buffer)