import numpy as np

from pyclad.models.adapters.pyod_adapters import (
    COPODAdapter,
    ECODAdapter,
    IsolationForestAdapter,
    LocalOutlierFactorAdapter,
    OneClassSVMAdapter,
)


def test_isolation_forest_smoke_run():
    model = IsolationForestAdapter(n_estimators=20, contamination=0.001)
    model.fit(np.array([[1, 2], [3, 4], [5, 6]]))
    model.predict(np.array([[1, 2], [3, 4], [5, 6]]))


def test_local_outlier_factor_smoke_run():
    model = LocalOutlierFactorAdapter(contamination=0.001, n_neighbors=2)
    model.fit(np.array([[1, 2], [3, 4], [5, 6]]))
    model.predict(np.array([[1, 2], [3, 4], [5, 6]]))


def test_one_class_svm_adapter_smoke_run():
    model = OneClassSVMAdapter(contamination=0.001)
    model.fit(np.array([[1, 2], [3, 4], [5, 6]]))
    model.predict(np.array([[1, 2], [3, 4], [5, 6]]))


def test_copod_smoke_run():
    model = COPODAdapter(contamination=0.001)
    model.fit(np.array([[1, 2], [3, 4], [5, 6]]))
    model.predict(np.array([[1, 2], [3, 4], [5, 6]]))


def test_ecod_smoke_run():
    model = ECODAdapter(contamination=0.001)
    model.fit(np.array([[1, 2], [3, 4], [5, 6]]))
    model.predict(np.array([[1, 2], [3, 4], [5, 6]]))
