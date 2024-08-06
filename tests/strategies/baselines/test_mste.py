from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.strategies.baselines.mste import MSTE
from tests.strategies.baselines.mock_model import MockModel


class MockModelForMSTE(MockModel):
    calls_count = 0

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, _: np.ndarray) -> (np.ndarray, np.ndarray):
        predictions = self.predictions[MockModelForMSTE.calls_count]
        MockModelForMSTE.calls_count += 1
        return predictions


@pytest.mark.parametrize("data", [([[1, 2, 3], [4, 5, 6]]), ([[1, 5, 8], [6, 1, 6], [1, 5, 3]])])
def test_creating_one_model_per_concept(data):
    model_fn = MagicMock(return_value=MockModel())
    strategy = MSTE(model_fn)
    for i, d in enumerate(data):
        strategy.learn(np.array(d), concept_id=i)
    assert model_fn.call_count == len(data)
    assert strategy.info()["strategy"]["number_of_models"] == len(data)


@pytest.mark.parametrize(
    "predictions",
    ([[[0, 0], [0.2, 0.4]], [[1, 1], [0.9, 0.8]]], [[[1, 0], [0.5, 0.8]], [[0, 1], [0.2, 0.7]], [[0, 0], [0.1, 0.2]]]),
)
def test_returning_predictions_from_the_right_model(predictions):
    MockModelForMSTE.calls_count = 0
    strategy = MSTE(lambda: MockModelForMSTE(predictions))
    for i, d in enumerate([[1, 1], [2, 2], [3, 3]]):
        strategy.learn(np.array(d), concept_id=i)

    for i, p in enumerate(predictions):
        results = strategy.predict(np.array([[1, 1], [1, 1], [1, 1]]), concept_id=i)
        assert_array_equal(results, p)


@pytest.mark.parametrize("data", [([[1, 2, 3], [4, 5, 6]]), ([[1, 5, 8], [6, 1, 6], [1, 5, 3]])])
def test_creating_correct_number_of_models(data):
    MockModelForMSTE.calls_count = 0
    strategy = MSTE(lambda: MockModel())
    for i, d in enumerate(data):
        strategy.learn(np.array(d), concept_id=i)

    assert strategy.info()["strategy"]["number_of_models"] == len(data)
