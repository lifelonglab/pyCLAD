import numpy as np

from pyclad.data import Concept, ConceptsDataset
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.naive import NaiveStrategy

set_seed(0)
train, test = [], []
for i, center in enumerate([0.0, 5.0]):
    train.append(Concept(f"c{i}", data=np.random.normal(center, 1.0, size=(200, 4))))
    test_data = np.concatenate([np.random.normal(center, 1.0, size=(50, 4)), np.random.normal(center + 3, 1.0, (5, 4))])
    test.append(Concept(f"c{i}", data=test_data, labels=np.array([0] * 50 + [1] * 5)))
dataset = ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)
strategy = NaiveStrategy(IsolationForestAdapter(random_state=0))

# --8<-- [start:builtin]
from pyclad.callbacks import ConceptMetricCallback, MemoryUsageCallback, TimeEvaluationCallback
from pyclad.metrics import AveragePrecision, BackwardTransfer, ContinualAverage, ForgettingMeasure, RocAuc

callbacks = [
    ConceptMetricCallback(
        base_metric=RocAuc(),
        summarized_metrics=[ContinualAverage(), BackwardTransfer()],
        stepwise_metrics=[ForgettingMeasure()],
    ),
    # One metric callback per base metric; each reports under its own key in the output.
    ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=[ContinualAverage()]),
    TimeEvaluationCallback(),
]
# tracemalloc slows the run down, so measure memory in a separate run from time.
memory_callbacks = [MemoryUsageCallback()]
# --8<-- [end:builtin]

# --8<-- [start:custom]
from pyclad.callbacks.callback import Callback
from pyclad.output.output_writer import InfoProvider


class ScoreRangeCallback(Callback, InfoProvider):
    """Records the range of anomaly scores on each evaluated concept after each learned concept."""

    def __init__(self):
        self._learned = None
        self._ranges = {}

    def after_training(self, learned_concept, *args, **kwargs):
        self._learned = learned_concept.name

    def after_evaluation(self, evaluated_concept, anomaly_scores, *args, **kwargs):
        key = f"{self._learned}->{evaluated_concept.name}"
        self._ranges[key] = [float(anomaly_scores.min()), float(anomaly_scores.max())]

    def info(self):
        return {"score_range_callback": self._ranges}


# --8<-- [end:custom]

callbacks.append(ScoreRangeCallback())
ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks).run()
memory_strategy = NaiveStrategy(IsolationForestAdapter(n_estimators=10, random_state=0))  # small: tracemalloc is slow
ConceptIncrementalScenario(dataset=dataset, strategy=memory_strategy, callbacks=memory_callbacks).run()
for callback in callbacks + memory_callbacks:
    print(callback.info())
