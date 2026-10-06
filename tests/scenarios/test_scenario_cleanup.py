import numpy as np
import pytest

from pyclad.callbacks import Callback
from pyclad.data import Concept, ConceptsDataset
from pyclad.output.prediction_results import PredictionResults
from pyclad.scenarios import (
    ConceptAgnosticScenario,
    ConceptAwareScenario,
    ConceptIncrementalScenario,
    SupervisedConceptIncrementalScenario,
)

SCENARIOS = [
    ConceptAgnosticScenario,
    ConceptAwareScenario,
    ConceptIncrementalScenario,
    SupervisedConceptIncrementalScenario,
]


class _Strategy:
    def __init__(self, fail_on_learn: bool = False):
        self._fail_on_learn = fail_on_learn

    def learn(self, *args, **kwargs) -> None:
        if self._fail_on_learn:
            raise RuntimeError("training failed")

    def predict(self, data: np.ndarray, **kwargs) -> PredictionResults:
        return PredictionResults(y_pred=np.zeros(len(data)), anomaly_scores=np.zeros(len(data)))


class _LifecycleRecorder(Callback):
    def __init__(self):
        self.events = []

    def before_scenario(self, *args, **kwargs):
        self.events.append("before_scenario")

    def after_scenario(self, *args, **kwargs):
        self.events.append("after_scenario")


class _FailingOnEvaluation(Callback):
    def after_evaluation(self, *args, **kwargs):
        raise ValueError("evaluation failed")


def _dataset() -> ConceptsDataset:
    concepts = [Concept(name, data=np.zeros((4, 2)), labels=np.array([0, 0, 1, 1])) for name in ["a", "b"]]
    return ConceptsDataset(name="dataset", train_concepts=concepts, test_concepts=concepts)


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_after_scenario_runs_once_on_success(scenario_class):
    recorder = _LifecycleRecorder()

    scenario_class(_dataset(), _Strategy(), [recorder]).run()

    assert recorder.events == ["before_scenario", "after_scenario"]


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_after_scenario_runs_when_training_fails(scenario_class):
    recorder = _LifecycleRecorder()

    with pytest.raises(RuntimeError, match="training failed"):
        scenario_class(_dataset(), _Strategy(fail_on_learn=True), [recorder]).run()

    assert recorder.events == ["before_scenario", "after_scenario"]


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_after_scenario_runs_when_a_callback_fails(scenario_class):
    recorder = _LifecycleRecorder()

    with pytest.raises(ValueError, match="evaluation failed"):
        scenario_class(_dataset(), _Strategy(), [_FailingOnEvaluation(), recorder]).run()

    assert recorder.events == ["before_scenario", "after_scenario"]
