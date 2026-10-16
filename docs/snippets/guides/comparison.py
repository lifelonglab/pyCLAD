import numpy as np

from pyclad.data import Concept, ConceptsDataset


def make_dataset() -> ConceptsDataset:
    """Three synthetic concepts; drawn after set_seed, so each seed gets its own data."""
    train, test = [], []
    for i, center in enumerate([0.0, 5.0, 10.0]):
        train.append(Concept(f"c{i}", data=np.random.normal(center, 1.0, size=(150, 4))))
        normal = np.random.normal(center, 1.0, size=(45, 4))
        anomalies = np.random.normal(center + 3, 1.0, size=(5, 4))
        test.append(Concept(f"c{i}", data=np.concatenate([normal, anomalies]), labels=np.array([0] * 45 + [1] * 5)))
    return ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)


# --8<-- [start:run]
import json
import pathlib

from pyclad.callbacks import ConceptMetricCallback
from pyclad.metrics import BackwardTransfer, ContinualAverage, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

# Every method gets the same model settings; only the strategy differs.
STRATEGIES = {
    "Naive": lambda: NaiveStrategy(IsolationForestAdapter()),
    "Replay": lambda: ReplayEnhancedStrategy(
        IsolationForestAdapter(), AdaptiveBalancedReplayBuffer(RandomSelection(), 100)
    ),
    "Cumulative": lambda: CumulativeStrategy(IsolationForestAdapter()),
}
SEEDS = [0, 1, 2]
results_dir = pathlib.Path("results")
results_dir.mkdir(exist_ok=True)

for name, make_strategy in STRATEGIES.items():
    for seed in SEEDS:
        seed_info = set_seed(seed)  # before creating the data, the model and the strategy
        dataset = make_dataset()
        strategy = make_strategy()
        callback = ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer()],
            on_undefined="propagate",  # record NaN instead of stopping; every run is reported
        )
        ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[callback]).run()
        JsonOutputWriter(results_dir / f"{name}_seed{seed}.json").write([seed_info, dataset, strategy, callback])
# --8<-- [end:run]

# --8<-- [start:aggregate]
import math

print(f"{'strategy':<12}{'metric':<20}{'mean':>8}{'std':>8}  runs")
for name in STRATEGIES:
    runs = [json.loads(path.read_text()) for path in sorted(results_dir.glob(f"{name}_seed*.json"))]
    for metric in ["ContinualAverage", "BackwardTransfer"]:
        values = [run["concept_metric_callback_ROC-AUC"]["metrics"][metric] for run in runs]
        values = [math.nan if value is None else value for value in values]  # NaN is saved as null
        undefined = sum(math.isnan(value) for value in values)
        print(f"{name:<12}{metric:<20}{np.mean(values):>8.3f}{np.std(values):>8.3f}  {len(values)} ({undefined} NaN)")
# --8<-- [end:aggregate]
