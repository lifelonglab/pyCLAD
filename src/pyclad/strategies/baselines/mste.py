from typing import Callable, Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import ConceptAwareStrategy


class MSTE(ConceptAwareStrategy):
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self._model_creation_fn = model_creation_fn
        self._models: Dict[str, Model] = {}  # concept_id: model

    def learn(self, data: np.ndarray, concept_id: str) -> None:
        new_model = self._model_creation_fn()
        new_model.fit(data)
        self._models[concept_id] = new_model

    def predict(self, data: np.ndarray, concept_id: str) -> (np.ndarray, np.ndarray):
        if concept_id in self._models:
            return self._models[concept_id].predict(data)
        else:
            return np.zeros(shape=data.shape[0]), np.zeros(shape=data.shape[0])

    def name(self) -> str:
        return "MSTE"

    def additional_info(self) -> Dict:
        return {
            "model_name": self._model_creation_fn().name(),
            "number_of_models": len(self._models),
        }
