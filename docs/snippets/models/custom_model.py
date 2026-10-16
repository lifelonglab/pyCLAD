import numpy as np

# --8<-- [start:model]
from sklearn.ensemble import IsolationForest

from pyclad.models.model import Model
from pyclad.output.prediction_results import PredictionResults


class SklearnIsolationForest(Model):
    """scikit-learn's Isolation Forest, converted to pyCLAD's conventions."""

    def __init__(self, n_estimators: int = 100, random_state: int | None = None):
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._model = IsolationForest(n_estimators=n_estimators, random_state=random_state)

    def fit(self, data: np.ndarray):
        self._model.fit(data)

    def predict(self, data: np.ndarray) -> PredictionResults:
        # scikit-learn: higher score_samples = more normal, predict() = +1 normal / -1 anomaly.
        # pyCLAD: higher anomaly score = more anomalous, 0 = normal / 1 = anomaly.
        return PredictionResults(
            y_pred=(self._model.predict(data) == -1).astype(int),
            anomaly_scores=-self._model.score_samples(data),
        )

    def name(self) -> str:
        return "SklearnIsolationForest"

    def additional_info(self):
        # Everything that shapes the result, so the output JSON tells two runs apart.
        return {"n_estimators": self._n_estimators, "random_state": self._random_state}


# --8<-- [end:model]

rng = np.random.default_rng(0)
model = SklearnIsolationForest(random_state=0)
model.fit(rng.normal(size=(300, 3)))
result = model.predict(np.concatenate([rng.normal(size=(5, 3)), rng.normal(8, 1, size=(5, 3))]))
assert result.anomaly_scores[5:].min() > result.anomaly_scores[:5].max()
assert result.y_pred[5:].all()
print(model.info())
