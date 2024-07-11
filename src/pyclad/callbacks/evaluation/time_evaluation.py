from typing import Any, Dict
import time

from pyclad.data.concept import Concept
from pyclad.callbacks.callback import Callback
from pyclad.output.output_writer import InfoProvider


class TimeEvaluationCallback(Callback, InfoProvider):
    def __init__(self):
        self._time_by_concept = dict()
        self._train_start = 0
        self._eval_start = 0
        self._train_time_total = 0
        self._eval_time_total = 0

    def before_training(self, *args, **kwargs):
        self._train_start = time.time()

    def after_training(self, learned_concept: Concept):
        train_time = time.time() - self._train_start
        self._time_by_concept[learned_concept.name] = dict()
        self._time_by_concept[learned_concept.name]["train_time"] = train_time
        self._train_time_total = self._train_time_total + train_time

    def before_evaluation(self, *args, **kwargs):
        self._eval_start = time.time()

    def after_evaluation(self, evaluated_concept: Concept, *args, **kwargs):
        eval_time = time.time() - self._eval_start
        self._eval_time_total = self._eval_time_total + eval_time

        if evaluated_concept.name in self._time_by_concept.keys():
            self._time_by_concept[evaluated_concept.name]["eval_time"] = eval_time
        else:
            self._time_by_concept[evaluated_concept.name] = dict()

    def info(self) -> Dict[str, Any]:

        return {
            "time_by_concept": self._time_by_concept,
            "train_time_total": self._train_time_total,
            "eval_time_total": self._eval_time_total,
        }
