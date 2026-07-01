import json
from typing import List

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.concepts_readers import read_concepts_from_df


class TabularCadDataset(ConceptsDataset):
    """
    Base class for the lifelonglab tabular continual anomaly detection (CAD) benchmarks hosted on Hugging Face
    <https://huggingface.co/collections/lifelonglab/tabular-cad-benchmarks>.

    Each benchmark is a single CSV with the columns ``task_id``, ``task_name``, ``task_split`` (``train``/``test``),
    ``label``, and the tabular features, accompanied by an ``orderings.json`` file that defines named task sequences
    for the continual scenario. The data is automatically downloaded and cached the first time it is used.

    Subclasses only declare the Hugging Face repository and a display name; the ``ordering`` parameter selects which
    task sequence from ``orderings.json`` is used to present the concepts (call :meth:`available_orderings` to list
    the valid values for a given benchmark).

    If using, please cite: <TODO>
    """

    _hf_repo: str
    _display_name: str
    _data_file: str = "data.csv"

    def __init__(self, ordering: str = "curriculum_asc", cache_dir: str = None):
        """
        :param ordering: Name of the task sequence from ``orderings.json`` used to order the concepts. Use
            :meth:`available_orderings` to list the valid values.
        :param cache_dir: Directory to cache the dataset. If `None`, the default cache directory is used.
        """
        task_sequence = self._load_ordering(self._hf_repo, ordering, cache_dir)

        data = load_dataset(self._hf_repo, data_files=self._data_file, cache_dir=cache_dir)
        df = data["train"].to_pandas().rename(columns={"task_id": "concept_id", "task_name": "concept_name"})

        train_df = df[df["task_split"] == "train"].drop(columns=["task_split"])
        test_df = df[df["task_split"] == "test"].drop(columns=["task_split"])

        train_concepts = self._reorder(read_concepts_from_df(train_df), task_sequence)
        test_concepts = self._reorder(read_concepts_from_df(test_df), task_sequence)

        super().__init__(
            name=f"{self._display_name}-{ordering}", train_concepts=train_concepts, test_concepts=test_concepts
        )

    @classmethod
    def available_orderings(cls, cache_dir: str = None) -> List[str]:
        """Return the names of the task sequences defined in the benchmark's ``orderings.json``."""
        return [o["name"] for o in cls._read_orderings(cls._hf_repo, cache_dir)]

    @classmethod
    def _load_ordering(cls, hf_repo: str, ordering: str, cache_dir: str) -> List[str]:
        orderings = cls._read_orderings(hf_repo, cache_dir)
        for o in orderings:
            if o["name"] == ordering:
                return o["task_sequence"]
        raise ValueError(f"Unknown ordering '{ordering}'. Available: {[o['name'] for o in orderings]}")

    @staticmethod
    def _read_orderings(hf_repo: str, cache_dir: str) -> List[dict]:
        path = hf_hub_download(repo_id=hf_repo, filename="orderings.json", repo_type="dataset", cache_dir=cache_dir)
        with open(path) as f:
            return json.load(f)["orderings"]

    @staticmethod
    def _reorder(concepts: List[Concept], task_sequence: List[str]) -> List[Concept]:
        by_name = {concept.name: concept for concept in concepts}
        ordered = [by_name[name] for name in task_sequence if name in by_name]
        leftovers = [concept for concept in concepts if concept.name not in set(task_sequence)]

        if len(leftovers) > 0:
            raise ValueError(f"Ordering skips the following concepts: {leftovers}")
        return ordered
