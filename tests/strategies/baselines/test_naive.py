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


def test_returning_model_predictions():
    from pyclad.output.prediction_results import PredictionResults
    expected = PredictionResults(y_pred=np.array([0, 1]), anomaly_scores=np.array([0.1, 0.9]))
    model = MockModel()
    model.predict = MagicMock(return_value=expected)
    strategy = NaiveStrategy(model)
    result = strategy.predict(np.array([[1, 1], [1, 1]]))
    assert_array_equal(result.y_pred, expected.y_pred)
    assert_array_equal(result.anomaly_scores, expected.anomaly_scores)
