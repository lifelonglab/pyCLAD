import abc

import numpy as np

from pyclad.data.concept import Concept
from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.strategy import Strategy


class SupervisedStrategy(Strategy):
    @abc.abstractmethod
    def learn(self, concept: Concept) -> None: ...

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> PredictionResults: ...
