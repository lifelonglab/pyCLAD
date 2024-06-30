import pathlib

import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset


def read_dataset_from_npy(filepath: pathlib.Path, dataset_name: str) -> ConceptsDataset:
    data = np.load(str(filepath), allow_pickle=True)

    train_concepts = []
    test_concepts = []

    for c in data:
        train_concepts.append(Concept(name=c["name"], data=c["train_data"], labels=None))
        if "test_data" in c and len(c["test_data"]) > 0:
            test_data = c["test_data"]
            test_labels = c["test_labels"]
            test_concepts.append(Concept(name=c["name"], data=test_data, labels=test_labels))

    return ConceptsDataset(name=dataset_name, train_concepts=train_concepts, test_concepts=test_concepts)
