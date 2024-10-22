from typing import List

from pyclad.callbacks.callback import Callback


class CallbackComposite(Callback):
    def __init__(self, callbacks: List[Callback]):
        self._callbacks = callbacks

    def before_scenario(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.before_scenario(*args, **kwargs)

    def after_scenario(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.after_scenario(*args, **kwargs)

    def before_training(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.before_training(*args, **kwargs)

    def after_training(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.after_training(*args, **kwargs)

    def before_evaluation(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.before_evaluation(*args, **kwargs)

    def after_evaluation(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.after_evaluation(*args, **kwargs)

    def before_concept_processing(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.before_concept_processing(*args, **kwargs)

    def after_concept_processing(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.after_concept_processing(*args, **kwargs)
