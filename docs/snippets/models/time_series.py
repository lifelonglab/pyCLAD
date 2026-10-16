import numpy as np

from pyclad.data import Concept, ConceptsDataset

rng = np.random.default_rng(0)
series = np.sin(np.linspace(0, 20, 400))[:, None] + rng.normal(0, 0.05, size=(400, 2))
labels = np.zeros(400)
dataset = ConceptsDataset(
    name="Series",
    train_concepts=[Concept("c0", data=series)],
    test_concepts=[Concept("c0", data=series, labels=labels)],
)

# --8<-- [start:windows]
from pyclad.data import convert_dataset_to_overlapping_windows
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.models.adapters.temporal_adapter import FlattenTimeSeriesAdapter

# Every concept becomes windows of shape (n_windows, 20, n_features); a window takes the label of its last step.
windowed = convert_dataset_to_overlapping_windows(window_size=20, dataset=dataset, step_size=5)

# Models expecting 2D input see each window flattened to one row.
model = FlattenTimeSeriesAdapter(IsolationForestAdapter(random_state=0))
# --8<-- [end:windows]

concept = windowed.train_concepts()[0]
model.fit(concept.data)
print(concept.data.shape, model.predict(concept.data).anomaly_scores.shape)
