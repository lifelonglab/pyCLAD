import numpy as np

from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import ContinualAverage, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptAgnosticScenario, ConceptAwareScenario, ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

set_seed(0)
train, test = [], []
for i, center in enumerate([0.0, 5.0, 10.0]):
    train.append(Concept(f"c{i}", data=np.random.normal(center, 1.0, size=(150, 4))))
    test_data = np.concatenate([np.random.normal(center, 1.0, size=(50, 4)), np.random.normal(center + 3, 1.0, (5, 4))])
    test.append(Concept(f"c{i}", data=test_data, labels=np.array([0] * 50 + [1] * 5)))
dataset = ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)

# --8<-- [start:scenarios]
# CumulativeStrategy implements all three contracts, so it can run in every scenario.
for scenario_cls in [ConceptAwareScenario, ConceptIncrementalScenario, ConceptAgnosticScenario]:
    strategy = CumulativeStrategy(IsolationForestAdapter(random_state=0))
    callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()])

    scenario_cls(dataset=dataset, strategy=strategy, callbacks=[callback]).run()
    print(scenario_cls.__name__, callback.info()["concept_metric_callback_ROC-AUC"]["metrics"])
# --8<-- [end:scenarios]
