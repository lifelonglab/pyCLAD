from typing import List

from pyclad.data.concept import Concept
from pyclad.data.dataset import Dataset


class ConceptsDataset(Dataset):
    def __init__(self, name: str, train_concepts: List[Concept], test_concepts: List[Concept]):
        self._train_concepts = train_concepts
        self._test_concepts = test_concepts
        self._name = name

    def train_concepts(self) -> List[Concept]:
        return self._train_concepts

    def test_concepts(self) -> List[Concept]:
        return self._test_concepts

    def name(self) -> str:
        return self._name

    def additional_info(self):
        return {"tran_concepts_no": len(self._train_concepts), "test_concepts_no": len(self._test_concepts)}
