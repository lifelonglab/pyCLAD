import pytest

from pyclad.strategies.regularization.lwf import LwFStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestLwFStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return LwFStrategy(model=backbone, alpha=0.5, batch_size=16, epochs=2, device="cpu")
