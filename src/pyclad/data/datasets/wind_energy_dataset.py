from typing import Literal

from datasets import load_dataset

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.concepts_readers import read_concepts_from_df


class WindEnergyDataset(ConceptsDataset):
    """
    Wind Energy dataset for continual anomaly detection. It was created leveraging this repository
    <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios>.
    See more details there and in the paper "Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights" here <https://ieeexplore.ieee.org/abstract/document/10473036/>.
    Please cite the paper provided above if using this dataset.
    """

    def __init__(
        self,
        dataset_type: Literal[
            "random_anomalies", "clustered_with_closest_assignment", "clustered_with_random_assignment"
        ],
        cache_dir: str = None,
    ):
        """
        :param dataset_type: The type of the dataset (see this repository <https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios> for more information).
        :param cache_dir: Directory to cache the dataset. If `None`, the default cache directory is used.
        """
        data = load_dataset(
            "lifelonglab/continual-wind-energy-anomaly-detection", data_dir=f"wind_{dataset_type}", cache_dir=cache_dir
        )
        train_concepts = read_concepts_from_df(data["train"].to_pandas())
        test_concepts = read_concepts_from_df(data["test"].to_pandas())
        super().__init__(name=f"Wind-Energy-{dataset_type}", train_concepts=train_concepts, test_concepts=test_concepts)
