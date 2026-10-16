import numpy as np

from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import ContinualAverage, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptAgnosticScenario
from pyclad.seed import set_seed
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory
from pyclad.strategies.vlad.vlad_strategy import VladConceptAgnosticStrategy

set_seed(0)
train, test = [], []
for i, center in enumerate([0.0, 5.0, 10.0]):
    train.append(Concept(f"c{i}", data=np.random.normal(center, 1.0, size=(300, 4))))
    test_data = np.concatenate([np.random.normal(center, 1.0, size=(50, 4)), np.random.normal(center + 3, 1.0, (5, 4))])
    test.append(Concept(f"c{i}", data=test_data, labels=np.array([0] * 50 + [1] * 5)))
dataset = ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)

# --8<-- [start:vlad]
# Build the memory with the factory matching the scenario: it picks a validated distance for it.
memory = VladMemory.for_concept_agnostic(memory_bound=600, min_distribution_size=128, max_comparison_size=128)
strategy = VladConceptAgnosticStrategy(model=IsolationForestAdapter(), memory=memory, max_steps_between_updates=5000)
# --8<-- [end:vlad]

callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()])
ConceptAgnosticScenario(dataset=dataset, strategy=strategy, callbacks=[callback]).run()
print(strategy.name(), callback.info()["concept_metric_callback_ROC-AUC"]["metrics"])
