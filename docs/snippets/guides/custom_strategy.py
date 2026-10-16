# --8<-- [start:strategy]
from typing import Any, Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.strategy import ConceptAgnosticStrategy, ConceptIncrementalStrategy


class SlidingWindowStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """Retrains the model on the most recent `window` samples.

    Needs neither concept boundaries nor concept identities, so it implements the incremental and agnostic
    contracts. It does not implement ConceptAwareStrategy: it would have no use for the concept id.
    """

    def __init__(self, model: Model, window: int):
        self._model = model
        self._window = window
        self._recent = None  # kept across learn() calls on purpose: the window spans concepts

    def learn(self, data: np.ndarray) -> None:
        self._recent = data if self._recent is None else np.concatenate([self._recent, data])
        self._recent = self._recent[-self._window :]
        self._model.fit(self._recent)

    def predict(self, data: np.ndarray) -> PredictionResults:
        return self._model.predict(data)  # must not change what was learned

    def name(self) -> str:
        return "SlidingWindow"

    def additional_info(self) -> Dict[str, Any]:
        return {"model": self._model.name(), "window": self._window}


# --8<-- [end:strategy]

# --8<-- [start:run]
from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import BackwardTransfer, ContinualAverage, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptAgnosticScenario, ConceptIncrementalScenario
from pyclad.seed import set_seed

set_seed(0)
train = [Concept(f"c{i}", data=np.random.normal(i * 5, 1, size=(300, 3))) for i in range(3)]
test = [
    Concept(
        f"c{i}",
        data=np.concatenate([np.random.normal(i * 5, 1, (45, 3)), np.random.normal(i * 5 + 4, 1, (5, 3))]),
        labels=np.array([0] * 45 + [1] * 5),
    )
    for i in range(3)
]
dataset = ConceptsDataset("Synthetic", train_concepts=train, test_concepts=test)

for scenario_cls in [ConceptIncrementalScenario, ConceptAgnosticScenario]:
    strategy = SlidingWindowStrategy(IsolationForestAdapter(random_state=0), window=500)
    callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage(), BackwardTransfer()])
    scenario_cls(dataset=dataset, strategy=strategy, callbacks=[callback]).run()
    print(scenario_cls.__name__, callback.info()["concept_metric_callback_ROC-AUC"]["metrics"])
# --8<-- [end:run]
