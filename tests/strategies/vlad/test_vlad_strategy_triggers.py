from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.strategies.vlad.change_point_detection.base import (
    ChangePoint,
    ChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory
from pyclad.strategies.vlad.vlad_strategy import VladStrategy
from tests.strategies.baselines.mock_model import MockModel


class ScriptedChangePointDetector(ChangePointDetector):
    """Test double returning a pre-scripted sequence of change points per call, so
    VladStrategy's retrain-trigger logic (Algorithm 1) can be tested independently of any real
    distance computation."""

    def __init__(self, script: List[List[ChangePoint]], memory: VladMemory):
        self._script = list(script)
        self._memory = memory

    def detect(self, batch: np.ndarray) -> List[ChangePoint]:
        return self._script.pop(0) if self._script else []

    @property
    def memory(self) -> VladMemory:
        return self._memory

    def name(self) -> str:
        return "Scripted"


def _memory(**overrides) -> VladMemory:
    params = dict(memory_bound=1000, summarization_trigger_ratio=1000.0, min_distribution_size=1000)
    params.update(overrides)
    return VladMemory(**params)


def _new_concept_cp() -> List[ChangePoint]:
    return [ChangePoint(index=0, is_new_concept=True, segment=np.zeros((1, 2)))]


def _recurring_cp() -> List[ChangePoint]:
    return [ChangePoint(index=0, is_new_concept=False, segment=np.zeros((1, 2)), matched_concept_id="concept-0")]


class TestRetrainTriggers:
    def test_first_call_always_trains_even_with_no_change_point(self):
        model = MockModel()
        model.fit = MagicMock()
        memory = _memory()
        strategy = VladStrategy(
            model, ScriptedChangePointDetector([[]], memory), memory, max_steps_between_updates=1000
        )

        strategy.learn(np.zeros((5, 2)))

        model.fit.assert_called_once()

    def test_no_change_and_within_time_budget_does_not_retrain(self):
        model = MockModel()
        model.fit = MagicMock()
        memory = _memory()
        strategy = VladStrategy(
            model, ScriptedChangePointDetector([[], []], memory), memory, max_steps_between_updates=1000
        )

        strategy.learn(np.zeros((5, 2)))  # first call: always trains
        model.fit.reset_mock()
        strategy.learn(np.zeros((5, 2)))  # no change, well within budget

        model.fit.assert_not_called()

    def test_new_concept_triggers_retrain(self):
        model = MockModel()
        model.fit = MagicMock()
        memory = _memory()
        script = [[], _new_concept_cp()]
        strategy = VladStrategy(
            model, ScriptedChangePointDetector(script, memory), memory, max_steps_between_updates=1000
        )

        strategy.learn(np.zeros((5, 2)))
        model.fit.reset_mock()
        strategy.learn(np.zeros((5, 2)))

        model.fit.assert_called_once()

    def test_recurring_concept_triggers_retrain(self):
        model = MockModel()
        model.fit = MagicMock()
        memory = _memory()
        script = [[], _recurring_cp()]
        strategy = VladStrategy(
            model, ScriptedChangePointDetector(script, memory), memory, max_steps_between_updates=1000
        )

        strategy.learn(np.zeros((5, 2)))
        model.fit.reset_mock()
        strategy.learn(np.zeros((5, 2)))

        model.fit.assert_called_once()

    def test_exceeding_time_budget_forces_retrain_without_any_change(self):
        model = MockModel()
        model.fit = MagicMock()
        memory = _memory()
        script = [[], [], []]
        strategy = VladStrategy(model, ScriptedChangePointDetector(script, memory), memory, max_steps_between_updates=8)

        strategy.learn(np.zeros((5, 2)))  # first call: always trains, resets counter
        model.fit.reset_mock()
        strategy.learn(np.zeros((5, 2)))  # steps_since_update = 5, within budget
        model.fit.assert_not_called()
        strategy.learn(np.zeros((5, 2)))  # steps_since_update = 10 > 8: forced retrain

        model.fit.assert_called_once()

    def test_retrain_uses_replay_buffer_concatenated_with_new_data(self):
        memory = _memory()
        memory.create_concept(np.full((5, 2), 7.0))
        model = MockModel()
        model.fit = MagicMock()
        strategy = VladStrategy(
            model, ScriptedChangePointDetector([[]], memory), memory, max_steps_between_updates=1000
        )

        new_data = np.zeros((3, 2))
        strategy.learn(new_data)

        called_with = model.fit.call_args.args[0]
        assert len(called_with) == 5 + 3
        assert_array_equal(called_with[:5], np.full((5, 2), 7.0))
        assert_array_equal(called_with[5:], new_data)


class TestAdditionalInfo:
    def test_does_not_duplicate_the_model_which_callers_already_log_separately(self):
        """The model is reported once, at the top level, by whoever calls
        output_writer.write([model, dataset, strategy, ...]) — VladStrategy must not re-embed it
        (that used to produce a redundant, doubly-nested {"model": {"model": {...}}})."""
        memory = _memory()
        strategy = VladStrategy(
            MockModel(), ScriptedChangePointDetector([[]], memory), memory, max_steps_between_updates=1000
        )

        info = strategy.additional_info()

        assert "model" not in info
        assert set(info.keys()) == {"change_point_detector", "memory", "max_steps_between_updates"}

    def test_change_point_detector_and_memory_info_are_reported_flat(self):
        memory = _memory()
        strategy = VladStrategy(
            MockModel(), ScriptedChangePointDetector([[]], memory), memory, max_steps_between_updates=1000
        )

        info = strategy.additional_info()

        assert info["change_point_detector"]["name"] == "Scripted"
        assert info["memory"]["name"] == "VladMemory"


class TestSharedMemoryValidation:
    def test_rejects_a_detector_wired_to_a_different_memory_instance(self):
        memory = _memory()
        other_memory = _memory()

        with pytest.raises(ValueError, match="same VladMemory instance"):
            VladStrategy(
                MockModel(),
                ScriptedChangePointDetector([[]], other_memory),
                memory,
                max_steps_between_updates=1000,
            )


if __name__ == "__main__":
    pytest.main([__file__])
