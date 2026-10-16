import numpy as np
import pandas as pd

# A CSV in the format the dataset below reads: one row per sample.
rng = np.random.default_rng(0)
rows = []
for concept_id, name in enumerate(["morning", "afternoon", "night"]):
    for split, n in [("train", 100), ("test", 40)]:
        labels = np.zeros(n) if split == "train" else np.array([0] * 36 + [1] * 4)
        features = rng.normal(concept_id * 3 + labels[:, None] * 4, 1.0, size=(n, 2))
        rows.append(
            pd.DataFrame(
                {
                    "concept_id": concept_id,
                    "concept_name": name,
                    "split": split,
                    "label": labels,
                    "f0": features[:, 0],
                    "f1": features[:, 1],
                }
            )
        )
pd.concat(rows).to_csv("sensors.csv", index=False)

# isort: split
# --8<-- [start:dataset]
import pandas as pd

from pyclad.data import ConceptsDataset
from pyclad.data.readers import read_concepts_from_df


class SensorDataset(ConceptsDataset):
    """Sensor readings in three operating modes.

    If you use this dataset, please cite: <the paper or data source>.
    """

    def __init__(self, path: str, train_anomalies: str = "drop"):
        """
        :param path: CSV with the columns concept_id, concept_name, split, label and the features.
        :param train_anomalies: "drop" removes labelled anomalies from the training data.
        """
        df = pd.read_csv(path)
        train_df = df[df["split"] == "train"].drop(columns=["split"])
        if train_anomalies == "drop":
            train_df = train_df[train_df["label"] == 0]
        test_df = df[df["split"] == "test"].drop(columns=["split"])
        self._train_anomalies = train_anomalies
        super().__init__(
            name="Sensors",
            # Concepts come out in concept_id order, which is the continual order.
            train_concepts=read_concepts_from_df(train_df),
            test_concepts=read_concepts_from_df(test_df),
        )

    def additional_info(self):
        return {**super().additional_info(), "train_anomalies": self._train_anomalies}


# --8<-- [end:dataset]

dataset = SensorDataset("sensors.csv")
print(dataset.info(), [c.name for c in dataset.train_concepts()])
assert [c.name for c in dataset.test_concepts()] == ["morning", "afternoon", "night"]

# --8<-- [start:hf]
from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class MyCadDataset(TabularCadDataset):
    """My benchmark. See TabularCadDataset for the format and the ``ordering`` parameter.

    If you use this dataset, please cite: <the paper>.
    """

    _hf_repo = "my-org/my-cad-benchmark"  # a CSV plus orderings.json, hosted on Hugging Face
    _display_name = "MyCAD"


# MyCadDataset(ordering="curriculum_asc", cache_dir=None) downloads it on first use.
# --8<-- [end:hf]
