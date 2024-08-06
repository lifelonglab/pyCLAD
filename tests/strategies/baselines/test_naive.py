from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.strategies.baselines.naive import NaiveStrategy
from tests.strategies.baselines.mock_model import MockModel


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
def test_learning_only_with_current_data(data):
    model = MockModel()
    mocked_fn = MagicMock()
    model.fit = mocked_fn
    strategy = NaiveStrategy(model)
    strategy.learn(data)
    mocked_fn.assert_called_with(data)


@pytest.mark.parametrize("data", [[[1, 2, 3], [4, 5, 6]], [[1, 5, 8], [6, 1, 6]]])
def test_returning_model_predictions(data):
    model = MockModel()
    mocked_fn = MagicMock(return_value=data)
    model.predict = mocked_fn
    strategy = NaiveStrategy(model)
    results = strategy.predict(np.array([[1, 1], [1, 1], [1, 1]]))
    assert_array_equal(results, data)
