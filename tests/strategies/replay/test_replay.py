from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy, ReplayOnlyStrategy
from tests.strategies.baselines.mock_model import MockModel


class ReplayBufferMock(ReplayBuffer):
    def update(self, data: np.ndarray) -> None:
        pass

    def data(self) -> np.ndarray:
        pass

    def name(self) -> str:
        return "ReplayBufferMock"


class TestReplayOnlyStrategy:
    @pytest.mark.parametrize(
        "data",
        [
            (
                np.array(
                    [[1, 2, 3], [4, 5, 6]],
                )
            ),
            (np.array([[1, 5, 8], [6, 1, 6]]),),
        ],
    )
    def test_training_model_with_data_from_replay_buffer(self, data):
        replay_buffer = ReplayBufferMock()
        replay_buffer.data = MagicMock(return_value=data)
        model = MockModel()
        model.fit = MagicMock()

        strategy = ReplayOnlyStrategy(model, replay_buffer)
        strategy.learn(np.array([[1, 1], [1, 1], [1, 1]]))

        model.fit.assert_called_with(data)

    @pytest.mark.parametrize(
        "data",
        [
            (
                np.array(
                    [[1, 2, 3], [4, 5, 6]],
                )
            ),
            (np.array([[1, 5, 8], [6, 1, 6]]),),
        ],
    )
    def test_returning_model_predictions(self, data):
        model = MockModel()
        mocked_fn = MagicMock(return_value=data)
        model.predict = mocked_fn

        strategy = ReplayOnlyStrategy(model, ReplayBufferMock())
        results = strategy.predict(np.array([[1, 1], [1, 1], [1, 1]]))

        assert_array_equal(results, data)

    @pytest.mark.parametrize(
        "data",
        [
            (
                np.array(
                    [[1, 2, 3], [4, 5, 6]],
                )
            ),
            (np.array([[1, 5, 8], [6, 1, 6]]),),
        ],
    )
    def test_adding_data_to_replay_buffer(self, data):
        replay_buffer = ReplayBufferMock()
        replay_buffer.update = MagicMock()
        model = MockModel()

        strategy = ReplayOnlyStrategy(model, replay_buffer)
        strategy.learn(data)

        replay_buffer.update.assert_called_with(data)


class TestReplayEnhancedStrategy:
    @pytest.mark.parametrize(
        "replay_data,new_data", [([[1, 2, 3], [4, 5, 6]], [[0, 1, 2]]), ([[1, 5, 8], [6, 1, 6]], [[1, 5, 3]])]
    )
    def test_training_model_with_current_data_and_replay_buffer(self, replay_data, new_data):
        replay_buffer = ReplayBufferMock()
        replay_buffer.data = MagicMock(return_value=np.array(replay_data))
        model = MockModel()
        model.fit = MagicMock()

        strategy = ReplayEnhancedStrategy(model, replay_buffer)
        strategy.learn(np.array(new_data))

        assert_array_equal(replay_data + new_data, model.fit.mock_calls[-1].args[0])

    @pytest.mark.parametrize("data", [(np.array([[1, 2, 3], [4, 5, 6]])), (np.array([[1, 5, 8], [6, 1, 6]]))])
    def test_returning_model_predictions(self, data):
        model = MockModel()
        mocked_fn = MagicMock(return_value=data)
        model.predict = mocked_fn

        strategy = ReplayEnhancedStrategy(model, ReplayBufferMock())
        results = strategy.predict(np.array([[1, 1], [1, 1], [1, 1]]))

        assert_array_equal(results, data)

    @pytest.mark.parametrize("data", [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 5, 8], [6, 1, 6]])])
    def test_adding_data_to_replay_buffer(self, data):
        replay_buffer = ReplayBufferMock()
        replay_buffer.update = MagicMock()
        replay_buffer.data = MagicMock(return_value=np.array([[0, 0, 0], [1, 1, 1]]))
        model = MockModel()

        strategy = ReplayEnhancedStrategy(model, replay_buffer)
        strategy.learn(data)

        replay_buffer.update.assert_called_with(data)
