from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyclad.models.classical.isolation_forest import IsolationForestAdapter
from pyclad.models.classical.lof import LocalOutlierFactorAdapter
from pyclad.models.classical.oc_svm import OneClassSVMAdapter


@patch("pyclad.models.classical.oc_svm.OneClassSVM")
@pytest.mark.parametrize("data", [[1, 2], [3, 4], [5, 6]])
def test_passing_data_to_model_while_learning(mock_if, data):
    mock_if.return_value.fit = MagicMock()
    model = OneClassSVMAdapter()

    model.learn(np.array(data))

    mock_if.return_value.fit.assert_called_once()
    assert_array_equal(mock_if.return_value.fit.mock_calls[0].args[0], data)


@patch("pyclad.models.classical.oc_svm.OneClassSVM")
@pytest.mark.parametrize("nu,gamma,kernel", [[0.5, 0.2, "auto"], [0.1, 0.3, "scale"], [0.01, 0.5, "auto"]])
def test_passing_parameters_to_model(mock_if, nu, gamma, kernel):
    OneClassSVMAdapter(nu=nu, kernel=kernel, gamma=gamma)

    assert mock_if.call_args.kwargs["nu"] == nu
    assert mock_if.call_args.kwargs["kernel"] == kernel
    assert mock_if.call_args.kwargs["gamma"] == gamma


@patch("pyclad.models.classical.oc_svm.OneClassSVM")
@pytest.mark.parametrize("labels,expected_labels,scores,expected_scores", [[1, 0, -0.5, 0.5], [-1, 1, 0.1, -0.1]])
def test_returning_adjusted_predictions_and_scores(mock_if, labels, expected_labels, scores, expected_scores):
    mock_if.return_value.predict = MagicMock(return_value=labels)
    mock_if.return_value.score_samples = MagicMock(return_value=np.array(scores))
    model = OneClassSVMAdapter()

    labels, scores = model.predict(np.array([[1, 2], [3, 4], [5, 6]]))
    assert_array_equal(labels, expected_labels)
    assert_array_equal(scores, expected_scores)


