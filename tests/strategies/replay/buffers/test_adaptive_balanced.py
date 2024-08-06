from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.selection.selection import SelectionMethod


class SelectionMethodMock(SelectionMethod):
    def select(self, data: np.ndarray, size: int) -> np.ndarray:
        pass

    def name(self) -> str:
        return "SelectionMethodMock"


@pytest.mark.parametrize("data", [[[1, 2, 3], [4, 5, 6]], [[1, 5, 8], [6, 1, 6]], [[1, 5, 3]]])
def test_putting_data_selected_by_selection_method_to_replay_buffer(data):
    selection_method = SelectionMethodMock()
    selection_method.select = MagicMock(return_value=np.array(data))

    buffer = AdaptiveBalancedReplayBuffer(selection_method, 10)
    buffer.update(np.array([[1, 2, 3], [4, 5, 6]]))

    assert_array_equal(buffer.data(), data)


@pytest.mark.parametrize(
    "buffer_size,input_sizes,expected_sizes", [(10, [20], [10]), (10, [20, 6], [5, 5]), (15, [20, 5, 100], [5, 5, 5])]
)
def test_balancing_available_size_between_all_buffers(buffer_size, input_sizes, expected_sizes):
    selection_method = SelectionMethodMock()
    selection_method.select = MagicMock(side_effect=lambda data, size: data[:size])

    buffer = AdaptiveBalancedReplayBuffer(selection_method, buffer_size)
    for i, size in enumerate(input_sizes):
        buffer.update(np.array([i] * size))

    for i, expected_size in enumerate(expected_sizes):
        assert buffer.data().tolist().count(i) == expected_size
