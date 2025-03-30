from typing import Literal

from datasets import load_dataset

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.concepts_readers import read_concepts_from_df


class NslKddDataset(ConceptsDataset):
    def __init__(
        self,
        dataset_type: Literal[
            "random_anomalies", "clustered_with_closest_assignment", "clustered_with_random_assignment"
        ],
        cache_dir: str = None,
    ):
        data = load_dataset("lifelonglab/continual-nsl-kdd", data_dir=f"nsl-kdd_{dataset_type}", cache_dir=cache_dir)
        train_concepts = read_concepts_from_df(data["train"].to_pandas())
        test_concepts = read_concepts_from_df(data["test"].to_pandas())
        super().__init__(name=f"NSL-KDD-{dataset_type}", train_concepts=train_concepts, test_concepts=test_concepts)


if __name__ == "__main__":
    dataset = NslKddDataset(dataset_type="random_anomalies")
    print(dataset.name())
