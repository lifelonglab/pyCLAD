import pytest

from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.regularization.ewc import EWCStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestEWCStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return EWCStrategy(backbone, StandardRunner(max_epochs=2), lambda_ewc=100, batch_size=16)


class TestEWCOnlineStrategy(BaseStrategyTest):
    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        return EWCStrategy(backbone, StandardRunner(max_epochs=2), lambda_ewc=100, batch_size=16, online=True)
