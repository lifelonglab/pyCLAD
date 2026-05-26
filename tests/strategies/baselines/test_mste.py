from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.baselines.mste import MSTE
from tests.strategies.baselines.mock_model import MockModel


class MockModelForMSTE(MockModel):
    calls_count = 0

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, _: np.ndarray) -> PredictionResults:
        result = self.predictions[MockModelForMSTE.calls_count]
        MockModelForMSTE.calls_count += 1
        return result


@pytest.mark.parametrize("data", [([[1, 2, 3], [4, 5, 6]]), ([[1, 5, 8], [6, 1, 6], [1, 5, 3]])])
def test_creating_one_model_per_concept(data):
    model_fn = MagicMock(return_value=MockModel())
    strategy = MSTE(model_fn)
    for i, d in enumerate(data):
        strategy.learn(np.array(d), concept_id=i)
    assert model_fn.call_count == len(data)
    assert strategy.info()["strategy"]["number_of_models"] == len(data)


def test_returning_predictions_from_the_right_model():
    MockModelForMSTE.calls_count = 0
    predictions = [
        PredictionResults(y_pred=np.array([0, 0]), anomaly_scores=np.array([0.2, 0.4])),
        PredictionResults(y_pred=np.array([1, 1]), anomaly_scores=np.array([0.9, 0.8])),
    ]
    strategy = MSTE(lambda: MockModelForMSTE(predictions))
    for i in range(2):
        strategy.learn(np.array([[1, 1]]), concept_id=i)

    for i, expected in enumerate(predictions):
        result = strategy.predict(np.array([[1, 1]]), concept_id=i)
        assert_array_equal(result.y_pred, expected.y_pred)
        assert_array_equal(result.anomaly_scores, expected.anomaly_scores)


def test_unknown_concept_returns_zeros():
    strategy = MSTE(lambda: MockModel())
    result = strategy.predict(np.array([[1, 1], [2, 2]]), concept_id="unseen")
    assert_array_equal(result.y_pred, np.zeros(2))
    assert_array_equal(result.anomaly_scores, np.zeros(2))


@pytest.mark.parametrize("data", [([[1, 2, 3], [4, 5, 6]]), ([[1, 5, 8], [6, 1, 6], [1, 5, 3]])])
def test_creating_correct_number_of_models(data):
    MockModelForMSTE.calls_count = 0
    strategy = MSTE(lambda: MockModel())
    for i, d in enumerate(data):
        strategy.learn(np.array(d), concept_id=i)

    assert strategy.info()["strategy"]["number_of_models"] == len(data)
