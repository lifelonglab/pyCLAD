import pytest

from pyclad.models.training.early_stopping import EarlyStopping
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.regularization.ewc import EWCStrategy
from tests.strategies.smoke_tests.base import BaseStrategyTest


class TestEWCWithEarlyStopping(BaseStrategyTest):
    """Runs the full concept-agnostic scenario with an early-stopping runner.

    Exercises the split -> run -> validate -> restore path inside the benchmark harness,
    not just the runner in isolation.
    """

    @pytest.fixture(scope="class")
    def strategy(self, backbone):
        runner = StandardRunner(max_epochs=8, validation_fraction=0.2, early_stopping=EarlyStopping(patience=1), seed=0)
        return EWCStrategy(backbone, runner, lambda_ewc=100, batch_size=16)
