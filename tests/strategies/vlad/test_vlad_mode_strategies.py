import pytest

from pyclad.strategies.vlad.change_point_detection.boundary_cpd import (
    BoundaryChangePointDetector,
)
from pyclad.strategies.vlad.change_point_detection.wasserstein_cpd import (
    WassersteinChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory
from pyclad.strategies.vlad.vlad_strategy import (
    VladConceptAgnosticStrategy,
    VladConceptIncrementalStrategy,
)
from tests.strategies.baselines.mock_model import MockModel


def _memory() -> VladMemory:
    return VladMemory(memory_bound=1000, min_distribution_size=8, mini_batch_size=4)


class TestVladConceptAgnosticStrategy:
    """VladConceptAgnosticStrategy exists so users don't have to import and wire a
    ChangePointDetector themselves — these tests just confirm it wires the right one."""

    def test_constructs_a_wasserstein_change_point_detector(self):
        memory = _memory()
        strategy = VladConceptAgnosticStrategy(MockModel(), memory, max_steps_between_updates=1000)

        assert isinstance(strategy._change_point_detector, WassersteinChangePointDetector)
        assert strategy._change_point_detector.memory is memory

    def test_forwards_refresh_every(self):
        memory = _memory()
        strategy = VladConceptAgnosticStrategy(MockModel(), memory, max_steps_between_updates=1000, refresh_every=7)

        assert strategy.additional_info()["change_point_detector"]["refresh_every"] == 7


class TestVladConceptIncrementalStrategy:
    def test_constructs_a_boundary_change_point_detector(self):
        memory = _memory()
        strategy = VladConceptIncrementalStrategy(MockModel(), memory, max_steps_between_updates=1000)

        assert isinstance(strategy._change_point_detector, BoundaryChangePointDetector)
        assert strategy._change_point_detector.memory is memory


if __name__ == "__main__":
    pytest.main([__file__])
