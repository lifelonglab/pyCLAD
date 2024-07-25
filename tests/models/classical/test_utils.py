import numpy as np
from numpy.testing import assert_array_equal

from pyclad.models.classical.utils import adjust_scikit_predictions


def test_adjust_scikit_predictions():
    predictions = np.array([1, -1, -1, 1, -1])
    adjusted_predictions = adjust_scikit_predictions(predictions)
    assert_array_equal(adjusted_predictions, np.array([0, 1, 1, 0, 1]))