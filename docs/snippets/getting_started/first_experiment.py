# --8<-- [start:imports]
import json
import pathlib

import numpy as np

from pyclad.analysis.scenario_heatmap import plot_metric_heatmap
from pyclad.callbacks import ConceptMetricCallback, TimeEvaluationCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import BackwardTransfer, ContinualAverage, ForgettingMeasure, ForwardTransfer, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.baselines.naive import NaiveStrategy

# --8<-- [end:imports]

# --8<-- [start:seed]
seed = set_seed(42)
# --8<-- [end:seed]


# --8<-- [start:concepts]
def make_concept(name: str, center: float):
    """Normal data around `center`; the test part also holds 10 anomalies far from it."""
    train = Concept(name, data=np.random.normal(center, 1.0, size=(300, 5)))

    normal = np.random.normal(center, 1.0, size=(90, 5))
    anomalies = np.random.normal(center + 6.0, 1.0, size=(10, 5))
    test = Concept(
        name,
        data=np.concatenate([normal, anomalies]),
        labels=np.concatenate([np.zeros(90), np.ones(10)]),  # 0 = normal, 1 = anomaly
    )
    return train, test


train1, test1 = make_concept("concept1", center=0.0)
train2, test2 = make_concept("concept2", center=10.0)
train3, test3 = make_concept("concept3", center=20.0)
# --8<-- [end:concepts]

# --8<-- [start:dataset]
dataset = ConceptsDataset(
    name="GeneratedDataset",
    train_concepts=[train1, train2, train3],  # the order in which the strategy learns them
    test_concepts=[test1, test2, test3],
)
# --8<-- [end:dataset]

# --8<-- [start:strategy]
model = IsolationForestAdapter(random_state=42)
strategy = NaiveStrategy(model)
# --8<-- [end:strategy]

# --8<-- [start:callbacks]
metric_callback = ConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
    stepwise_metrics=[ForgettingMeasure()],
)
time_callback = TimeEvaluationCallback()
# --8<-- [end:callbacks]

# --8<-- [start:scenario]
scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback, time_callback])
scenario.run()
# --8<-- [end:scenario]

# --8<-- [start:output]
output_writer = JsonOutputWriter(pathlib.Path("output.json"))
output_writer.write([seed, model, dataset, strategy, metric_callback, time_callback])
# --8<-- [end:output]

# --8<-- [start:results]
with open("output.json") as f:
    results = json.load(f)

roc_auc = results["concept_metric_callback_ROC-AUC"]
print(roc_auc["metrics"])  # ContinualAverage, BackwardTransfer, ForwardTransfer
print(roc_auc["stepwise_metrics"])  # ForgettingMeasure after each learned concept

plot_metric_heatmap(
    roc_auc["metric_matrix"],
    roc_auc["concepts_order"],
    annotate=True,
    output_path=pathlib.Path("heatmap.png"),
)
# --8<-- [end:results]
