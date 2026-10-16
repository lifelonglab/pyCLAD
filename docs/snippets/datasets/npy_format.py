import pathlib

import numpy as np

# --8<-- [start:format]
# One dict per concept: train data (normal only), and test data with labels (0 = normal, 1 = anomaly).
rng = np.random.default_rng(0)
concepts = np.array(
    [
        {
            "name": f"concept{i}",
            "train_data": rng.normal(i, 1.0, size=(100, 3)),
            "test_data": rng.normal(i, 1.0, size=(20, 3)),
            "test_labels": np.array([0] * 18 + [1] * 2),
        }
        for i in range(3)
    ],
    dtype=object,
)
np.save("my_scenario.npy", concepts, allow_pickle=True)
# --8<-- [end:format]

# --8<-- [start:read]
from pyclad.data.readers import read_dataset_from_npy

dataset = read_dataset_from_npy(pathlib.Path("my_scenario.npy"), dataset_name="MyScenario")
# --8<-- [end:read]
assert [c.name for c in dataset.test_concepts()] == ["concept0", "concept1", "concept2"]
print(dataset.info())
