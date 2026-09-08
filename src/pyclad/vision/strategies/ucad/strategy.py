from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy
from pyclad.vision.models.ucad.ucad import UCAD
from pyclad.vision.prediction_results import VisionPredictionResults


class UCADStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Continual strategy for UCAD's key-prompt-knowledge memory (Liu et al., AAAI 2024).

    The continual mechanism lives in the model, which appends one memory entry per concept and
    never resets. This adapter exists for one reason on each side: ``learn`` passes the concept
    id down so the model can find that concept's SAM structure masks, and ``predict``
    **discards** the concept id it is handed so that task identity can only come from the
    learned keys.
    """

    def __init__(self, model: UCAD):
        self._model = model
        self._tasks_seen = 0

    def learn(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> None:
        del kwargs
        self._model.set_current_concept(concept_id)
        self._model.fit(data)
        self._tasks_seen += 1

    def predict(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> VisionPredictionResults:
        # concept_id is deliberately discarded: revealing the evaluated concept would turn a
        # task-agnostic method into a task-aware one and inflate every reported metric.
        del concept_id, kwargs
        return self._model.predict(data)

    def name(self) -> str:
        return "UCAD"

    def additional_info(self) -> Dict:
        return {
            "continual_mechanism": "key_prompt_knowledge",
            "task_agnostic_inference": True,
            "tasks_seen": self._tasks_seen,
        }
