import numpy as np
from pyod.models.base import BaseDetector
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from pyclad.models.model import Model


class PyODAdapter(Model):
    def __init__(self, model: BaseDetector, model_name: str):
        self._model = model
        self._model_name = model_name

    def fit(self, data: np.ndarray):
        self._model.fit(data)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._model.predict(data), self._model.decision_function(data)

    def name(self) -> str:
        return self._model_name

    def additional_info(self):
        return self._model.get_params()


class IsolationForestAdapter(PyODAdapter):
    def __init__(self, **kwargs):
        super().__init__(model_name="IsolationForest", model=IForest(**kwargs))


class LocalOutlierFactorAdapter(PyODAdapter):
    def __init__(self, **kwargs):
        super().__init__(model_name="LOF", model=LOF(novelty=True, **kwargs))


class OneClassSVMAdapter(PyODAdapter):
    def __init__(self, **kwargs):
        super().__init__(model_name="OneClassSVM", model=OCSVM(**kwargs))


class COPODAdapter(PyODAdapter):
    def __init__(self, contamination=0.00001):
        super().__init__(model_name="COPOD", model=COPOD(contamination=contamination))


class ECODAdapter(PyODAdapter):
    def __init__(self, contamination=0.00001):
        super().__init__(model_name="ECOD", model=ECOD(contamination=contamination))
