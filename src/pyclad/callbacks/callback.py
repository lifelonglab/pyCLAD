import abc

import numpy as np


class Callback(abc.ABC):
    def before_scenario(self, *args, **kwargs):
        pass

    def after_scenario(self, *args, **kwargs):
        pass

    def before_training(self, *args, **kwargs):
        pass

    def after_training(self, *args, **kwargs):
        pass

    def before_evaluation(self, *args, **kwargs):
        pass

    def after_evaluation(self, *args, **kwargs):
        pass


