from numpy.testing import assert_array_equal

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset


def test_returning_correct_concepts():
    train_concepts = [
        Concept(name="concept1", data=[1, 2, 3], labels=None),
        Concept(name="concept2", data=[4, 5, 6], labels=None),
    ]
    test_concepts = [
        Concept(name="concept1", data=[10, 20, 30], labels=[1, 1, 0]),
        Concept(name="concept2", data=[40, 50, 60], labels=[0, 0, 0]),
    ]
    dataset = ConceptsDataset(name="test_dataset", train_concepts=train_concepts, test_concepts=test_concepts)

    assert_array_equal(train_concepts, dataset.train_concepts())
    assert_array_equal(test_concepts, dataset.test_concepts())
