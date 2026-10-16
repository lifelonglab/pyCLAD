import numpy as np

# --8<-- [start:pyod]
from pyod.models.knn import KNN

from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter, PyODAdapter

# Ready-made adapters pass their keyword arguments to the PyOD detector.
iforest = IsolationForestAdapter(n_estimators=200, random_state=0)

# Any other PyOD detector can be wrapped directly.
knn = PyODAdapter(KNN(n_neighbors=10), model_name="KNN")
# --8<-- [end:pyod]

# --8<-- [start:predict]
train = np.random.default_rng(0).normal(size=(200, 4))
iforest.fit(train)
result = iforest.predict(train[:5])
print(result.anomaly_scores)  # higher means more anomalous
print(result.y_pred)  # 0 = normal, 1 = anomaly
# --8<-- [end:predict]

knn.fit(train)
print(knn.info())
