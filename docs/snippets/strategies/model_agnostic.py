import numpy as np

from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import BackwardTransfer, ContinualAverage, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptAwareScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.mste import MSTE
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy, ReplayOnlyStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

set_seed(0)
train, test = [], []
for i, center in enumerate([0.0, 5.0, 10.0]):
    train.append(Concept(f"c{i}", data=np.random.normal(center, 1.0, size=(150, 4))))
    test_data = np.concatenate([np.random.normal(center, 1.0, size=(50, 4)), np.random.normal(center + 3, 1.0, (5, 4))])
    test.append(Concept(f"c{i}", data=test_data, labels=np.array([0] * 50 + [1] * 5)))
dataset = ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)

# --8<-- [start:strategies]
strategies = [
    NaiveStrategy(IsolationForestAdapter()),
    CumulativeStrategy(IsolationForestAdapter()),
    # MSTE creates a new model for every concept, so it takes a function rather than a model.
    MSTE(lambda: IsolationForestAdapter()),
    ReplayOnlyStrategy(
        IsolationForestAdapter(),
        AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=200),
    ),
    ReplayEnhancedStrategy(
        IsolationForestAdapter(),
        AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=200),
    ),
]
# --8<-- [end:strategies]

for strategy in strategies:
    callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage(), BackwardTransfer()])
    ConceptAwareScenario(dataset=dataset, strategy=strategy, callbacks=[callback]).run()
    print(strategy.name(), callback.info()["concept_metric_callback_ROC-AUC"]["metrics"])
