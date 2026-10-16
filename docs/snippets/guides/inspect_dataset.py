import pathlib

from pyclad.data.readers import read_dataset_from_npy

# The NSL-KDD scenario shipped with the repository's examples.
path = (
    pathlib.Path(__file__).parents[3]
    / "examples"
    / "resources"
    / "nsl-kdd_random_anomalies_5_concepts_1000_per_cluster.npy"
)
dataset = read_dataset_from_npy(path, dataset_name="NSL-KDD-R")

# --8<-- [start:inspect]
import numpy as np


def describe(dataset):
    """One line per concept: sizes, anomaly share, and whether ROC-AUC will be defined."""
    test_by_name = {concept.name: concept for concept in dataset.test_concepts()}
    for train in dataset.train_concepts():
        test = test_by_name.get(train.name)
        line = f"{train.name:<12} train={len(train.data):>6}"
        if test is None:
            line += "  no test concept"
        else:
            anomalies = int(np.sum(test.labels == 1))
            line += f"  test={len(test.data):>6}  anomalies={anomalies / len(test.labels):6.1%}"
            if anomalies in (0, len(test.labels)):
                line += "  single class: ROC-AUC undefined"
        print(line)


describe(dataset)
# --8<-- [end:inspect]

assert len(dataset.train_concepts()) == 5
