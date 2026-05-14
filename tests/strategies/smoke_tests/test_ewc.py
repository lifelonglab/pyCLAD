import pytest

from pyclad.strategies.regularization.ewc import EWCStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestEWCStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return EWCStrategy(backbone, lambda_ewc=100, epochs=2, batch_size=16)