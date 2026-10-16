# requires: torch
import runpy
from pathlib import Path

# In the repository the strategy would be imported; here it comes from the guide's strategy snippet.
SlidingWindowStrategy = runpy.run_path(str(Path(__file__).with_name("custom_strategy.py")))["SlidingWindowStrategy"]

# --8<-- [start:test]
# tests/strategies/smoke_tests/test_sliding_window.py
import pytest

from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.training.runners.standard import StandardRunner
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestSlidingWindowStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return SlidingWindowStrategy(
            TorchModelAdapter(backbone, StandardRunner(max_epochs=2), batch_size=16), window=100
        )


# --8<-- [end:test]

assert issubclass(TestSlidingWindowStrategy, BaseStrategyTest)
