import math

import numpy as np

from pyclad.data import Concept, ConceptsDataset
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.naive import NaiveStrategy

set_seed(0)
dataset = ConceptsDataset(
    name="OneConceptWithoutAnomalies",
    train_concepts=[
        Concept("c0", data=np.random.normal(size=(100, 3))),
        Concept("c1", np.random.normal(size=(100, 3))),
    ],
    test_concepts=[
        Concept("c0", data=np.random.normal(size=(20, 3)), labels=np.array([0] * 18 + [1] * 2)),
        Concept("c1", data=np.random.normal(size=(20, 3)), labels=np.zeros(20)),  # no anomaly: ROC-AUC is undefined
    ],
)

# --8<-- [start:propagate]
from pyclad.callbacks import ConceptMetricCallback
from pyclad.metrics import ContinualAverage, RocAuc

callback = ConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[ContinualAverage()],
    on_undefined="propagate",  # the default, "raise", would stop the run at the first undefined cell
)
# --8<-- [end:propagate]

ConceptIncrementalScenario(
    dataset=dataset, strategy=NaiveStrategy(IsolationForestAdapter()), callbacks=[callback]
).run()
result = callback.info()["concept_metric_callback_ROC-AUC"]
print(result["undefined_concepts"], result["metrics"])
assert result["undefined_concepts"] == ["c1"] and math.isnan(result["metrics"]["ContinualAverage"])
