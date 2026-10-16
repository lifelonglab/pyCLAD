import numpy as np

from pyclad.data import Concept, ConceptsDataset
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.naive import NaiveStrategy

set_seed(0)
names = ["bottle", "cable", "capsule", "carpet", "grid"]
train, test = [], []
for i, name in enumerate(names):
    train.append(Concept(name, data=np.random.normal(3 * i, 1.0, size=(100, 4))))
    test_data = np.concatenate([np.random.normal(3 * i, 1.0, size=(40, 4)), np.random.normal(3 * i + 2, 1.0, (4, 4))])
    test.append(Concept(name, data=test_data, labels=np.array([0] * 40 + [1] * 4)))
dataset = ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)

# --8<-- [start:schedule]
from pyclad.data import apply_step_schedule

scheduled = apply_step_schedule(dataset, schedule="3-2")  # 5 concepts -> 2 training steps

print([c.name for c in scheduled.train_concepts()])  # ['step_0', 'step_1']
print([c.name for c in scheduled.test_concepts()])  # still the 5 concepts
print(scheduled.first_seen_step())  # {'bottle': 0, 'cable': 0, 'capsule': 0, 'carpet': 1, 'grid': 1}
# --8<-- [end:schedule]
assert len(scheduled.train_concepts()) == 2 and len(scheduled.test_concepts()) == 5

# --8<-- [start:metrics]
from pyclad.callbacks import ScheduleAwareConceptMetricCallback
from pyclad.metrics import (
    FinalStepAverage,
    RocAuc,
    ScheduleAwareForgettingMeasure,
    ScheduleAwareForwardTransfer,
    ScheduleAwareNewTaskAcquisition,
)

callback = ScheduleAwareConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[FinalStepAverage()],  # square-only metrics such as ContinualAverage would raise
    schedule_aware_metrics=[
        ScheduleAwareForgettingMeasure(),
        ScheduleAwareForwardTransfer(),
        ScheduleAwareNewTaskAcquisition(),
    ],
    first_seen_step=scheduled,  # the grouped dataset carries the mapping
)
# --8<-- [end:metrics]

ConceptIncrementalScenario(
    dataset=scheduled, strategy=NaiveStrategy(IsolationForestAdapter()), callbacks=[callback]
).run()
print(callback.info())
