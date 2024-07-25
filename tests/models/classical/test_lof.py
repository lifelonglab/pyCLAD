from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.models.classical.isolation_forest import IsolationForestAdapter
from pyclad.models.classical.lof import LocalOutlierFactorAdapter


@patch("pyclad.models.classical.lof.LocalOutlierFactor")
@pytest.mark.parametrize("data", [[1, 2], [3, 4], [5, 6]])
def test_passing_data_to_model_while_learning(mock_if, data):
    mock_if.return_value.fit = MagicMock()
    model = LocalOutlierFactorAdapter(n_neighbors=10)

    model.learn(np.array(data))

    mock_if.return_value.fit.assert_called_once()
    assert_array_equal(mock_if.return_value.fit.mock_calls[0].args[0], data)


@patch("pyclad.models.classical.lof.LocalOutlierFactor")
@pytest.mark.parametrize("n_neighbors", [10, 100, 5])
def test_passing_parameters_to_model(mock_if, n_neighbors):
    LocalOutlierFactorAdapter(n_neighbors=n_neighbors)

    assert mock_if.call_args.kwargs["n_neighbors"] == n_neighbors


@patch("pyclad.models.classical.lof.LocalOutlierFactor")
@pytest.mark.parametrize("labels,expected_labels,scores,expected_scores", [[1, 0, -0.5, 0.5], [-1, 1, 0.1, -0.1]])
def test_returning_adjusted_predictions_and_scores(mock_if, labels, expected_labels, scores, expected_scores):
    mock_if.return_value.predict = MagicMock(return_value=labels)
    mock_if.return_value.score_samples = MagicMock(return_value=np.array(scores))
    model = LocalOutlierFactorAdapter()

    labels, scores = model.predict(np.array([[1, 2], [3, 4], [5, 6]]))
    assert_array_equal(labels, expected_labels)
    assert_array_equal(scores, expected_scores)


