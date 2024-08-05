import numpy as np
import pytest

from pyclad.strategies.replay.selection.random import RandomSelection


@pytest.mark.parametrize("input_size, selection_size", [(100, 10), (80, 50), (100, 100), (10, 20)])
def test_selecting_expected_size(input_size, selection_size):
    rng = np.random.default_rng(seed=42)
    input_data = rng.random((input_size, 3))

    selection_method = RandomSelection()
    selected_data = selection_method.select(input_data, selection_size)

    assert selected_data.shape[0] == min(selection_size, input_size)
