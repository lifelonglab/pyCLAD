import abc


class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, anomaly_scores, y_pred, y_true) -> float: ...

    """ Compute the metric.
    Args:
        anomaly_scores: The anomaly scores.
        y_pred: The predicted labels.
        y_true: The true labels.
    Returns:
        The computed metric."""

    @abc.abstractmethod
    def name(self) -> str: ...

    """ Get the name of the metric."""
