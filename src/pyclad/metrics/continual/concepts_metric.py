import abc
from typing import List


MetricMatrix = List[List[float]]


class ConceptsMatrixMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, metric_matrix: MetricMatrix) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...


ConceptLevelMatrix = List[List[float]]


class ConceptLevelMatrixMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...
