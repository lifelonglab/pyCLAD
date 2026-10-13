from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from tests.strategies.baselines.mock_model import MockModel


@pytest.mark.parametrize("data", [([[4, 5, 6], [1, 2, 3]],), ([[1, 5, 8], [6, 1, 6]],)])
def test_learning_with_all_data(data):
    model = MockModel()
    mocked_fn = MagicMock()
    MockModel.fit = mocked_fn
    strategy = CumulativeStrategy(model)
    for d in data:
        d = np.array(d)
        strategy.learn(d)
        assert_array_equal(d, mocked_fn.mock_calls[-1].args[0])


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
def test_returning_model_predictions(data):
    model = MockModel()
    mocked_fn = MagicMock(return_value=data)
    model.predict = mocked_fn
    strategy = CumulativeStrategy(model)
    results = strategy.predict(np.array([[1, 1], [1, 1], [1, 1]]))
    assert_array_equal(results, data)


def test_info_reports_buffer_size_before_and_after_learning():
    strategy = CumulativeStrategy(MockModel())
    assert strategy.info()["strategy"]["buffer_size"] == 0

    strategy.learn(np.array([[1, 2], [3, 4]]))
    strategy.learn(np.array([[5, 6]]))
    assert strategy.info()["strategy"]["buffer_size"] == 3
