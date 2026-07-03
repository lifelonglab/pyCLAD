import pytest

from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.regularization.lwf import LwFStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestLwFStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return LwFStrategy(model=backbone, runner=StandardRunner(max_epochs=2), alpha=0.5, batch_size=16, device="cpu")
