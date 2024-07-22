import abc
from typing import List

ResultsMatrix = List[List[float]]


class ConceptsResultsMatrix(abc.ABC):
    @abc.abstractmethod
    def compute(self, metric_matrix: ResultsMatrix) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...


ConceptLevelMatrix = List[List[float]]


class ConceptLevelMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...
