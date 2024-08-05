import time
from collections import defaultdict
from typing import Any, Dict

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.output.output_writer import InfoProvider


class TimeEvaluationCallback(Callback, InfoProvider):
    def __init__(self):
        self._time_by_concept = defaultdict(lambda: dict({"train_time": 0, "eval_time": 0}))
        self._train_start = 0
        self._eval_start = 0
        self._train_time_total = 0
        self._eval_time_total = 0

    def before_training(self, *args, **kwargs):
        self._train_start = time.time()

    def after_training(self, learned_concept: Concept):
        train_time = time.time() - self._train_start
        self._time_by_concept[learned_concept.name]["train_time"] = train_time
        self._train_time_total = self._train_time_total + train_time

    def before_evaluation(self, *args, **kwargs):
        self._eval_start = time.time()

    def after_evaluation(self, evaluated_concept: Concept, *args, **kwargs):
        eval_time = time.time() - self._eval_start
        self._eval_time_total = self._eval_time_total + eval_time
        self._time_by_concept[evaluated_concept.name]["eval_time"] += eval_time

    def info(self) -> Dict[str, Any]:
        return {
            "time_evaluation_callback": {
                "time_by_concept": self._time_by_concept,
                "train_time_total": self._train_time_total,
                "eval_time_total": self._eval_time_total,
            }
        }
