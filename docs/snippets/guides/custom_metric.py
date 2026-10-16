import math

import numpy as np

# --8<-- [start:base]
from pyclad.metrics import BaseMetric


class PrecisionAtK(BaseMetric):
    """Share of true anomalies among the k samples with the highest anomaly scores."""

    def __init__(self, k: int):
        self._k = k

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_true = np.asarray(y_true)
        if y_true.sum() == 0 or len(y_true) < self._k:
            return float("nan")  # undefined: never substitute 0 or 0.5
        top_k = np.argsort(-np.asarray(anomaly_scores))[: self._k]  # higher score = more anomalous
        return float(y_true[top_k].mean())

    def name(self) -> str:
        return f"Precision@{self._k}"  # also names the callback's key in the output


# --8<-- [end:base]

# --8<-- [start:summarized]
from pyclad.metrics import ConceptLevelMatrix, SummarizedMetric
from pyclad.metrics.continual.concepts_metric import validate_square_matrix


class FinalStepWorstConcept(SummarizedMetric):
    """The lowest score among earlier concepts after the last training step: the most forgotten concept."""

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())  # it reads the diagonal structure
        earlier = metric_matrix[-1][:-1] if len(metric_matrix) > 1 else []
        # min() of a list holding NaN is not NaN, so check explicitly: an undefined cell makes the result NaN.
        if not earlier or any(math.isnan(value) for value in earlier):
            return float("nan")
        return float(min(earlier))

    def name(self) -> str:
        return "FinalStepWorstConcept"


# --8<-- [end:summarized]


# --8<-- [start:tests]
def test_precision_at_k_on_a_hand_computed_example():
    scores = [0.9, 0.8, 0.1, 0.7]
    y_true = [1, 0, 1, 0]
    assert PrecisionAtK(k=2).compute(scores, None, y_true) == 0.5  # top 2: samples 0 (anomaly) and 1 (normal)


def test_precision_at_k_is_nan_without_anomalies():
    assert math.isnan(PrecisionAtK(k=2).compute([0.1, 0.2, 0.3], None, [0, 0, 0]))


def test_worst_concept_on_a_hand_computed_matrix():
    matrix = [[0.9, 0.5, 0.5], [0.8, 0.9, 0.5], [0.6, 0.8, 0.9]]
    assert FinalStepWorstConcept().compute(matrix) == 0.6  # last row, earlier concepts: min(0.6, 0.8)


def test_worst_concept_propagates_nan():
    assert math.isnan(FinalStepWorstConcept().compute([[0.9, 0.5], [math.nan, 0.9]]))


# --8<-- [end:tests]

for test in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
    test()
