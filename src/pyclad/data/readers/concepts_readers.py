import pathlib

import numpy as np
import pandas as pd

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


def read_concepts_from_df(df: pd.DataFrame) -> list[Concept]:
    """
    Read concept from pd.DataFrame that follows the following schema:
    - concept_id: int - needs to be continuous between df['concept_id'].min() and df['concept_id'].max()
    - concept_name: str - the same for all rows with the same concept_id
    - label: Any
    - other columns - data for the concept
    """
    if "concept_id" not in df.columns or "concept_name" not in df.columns or "label" not in df.columns:
        raise ValueError("Concepts DataFrame should have 'concept_id', 'concept_name', and 'label' columns")
    concepts = []

    min_concept_id = df["concept_id"].min()
    max_concept_id = df["concept_id"].max()

    if not np.array_equal(np.sort(df["concept_id"].unique()), np.arange(min_concept_id, max_concept_id + 1)):
        raise ValueError(f"Concept ids should start from 0 and be continuous, but got {df['concept_id'].unique()}")

    for i in range(min_concept_id, max_concept_id + 1):
        concept_df = df[df["concept_id"] == i]
        concept_name = concept_df["concept_name"].values[0]
        concept_labels = concept_df["label"].values
        concept_data = concept_df.drop(columns=["concept_id", "concept_name", "label"]).values
        concepts.append(Concept(name=concept_name, data=concept_data, labels=concept_labels))

    return concepts
