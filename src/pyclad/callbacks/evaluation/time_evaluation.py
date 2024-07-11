from typing import Any, Dict
import time

from pyclad.callbacks.callback import Callback
from pyclad.output.output_writer import InfoProvider


class TimeEvaluationCallback(Callback, InfoProvider):
    def __init__(self):
        self._train_start = 0
        self._train_end = 0
        self._train_time = 0
        self._eval_start = 0
        self._eval_end = 0
        self._eval_time = 0

    def before_training(self, *args, **kwargs):
        self._train_start = time.time()

    def after_training(self, *args, **kwargs):
        self._train_end = time.time()
        self._train_time = self._train_end - self._train_start

    def before_evaluation(self, *args, **kwargs):
        self._eval_start = time.time()

    def after_evaluation(self, *args, **kwargs):
        self._eval_end = time.time()
        self._eval_time = self._eval_end - self._eval_start

    def info(self) -> Dict[str, Any]:

        return {
            "train_time": self._train_time,
            "eval_time": self._eval_time
        }
