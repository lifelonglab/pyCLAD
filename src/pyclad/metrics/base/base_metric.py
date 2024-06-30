import abc


class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, anomaly_scores, y_pred, y_true) -> float: ...

    @abc.abstractmethod
    def name(self) -> str: ...
