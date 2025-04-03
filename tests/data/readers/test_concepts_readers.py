import numpy as np
import pandas as pd
import pytest

from pyclad.data.readers.concepts_readers import read_concepts_from_df

sample_df = pd.DataFrame(
    {
        "concept_id": [0, 0, 1, 1],
        "concept_name": ["name1", "name1", "name2", "name2"],
        "label": [0, 0, 1, 0],
        "feature_1": [1, 2, 3, 4],
        "feature_2": [5, 6, 7, 8],
    }
)


def test_read_concepts_from_df():
    df = sample_df.copy()
    concepts = read_concepts_from_df(df)

    assert len(concepts) == 2
    assert concepts[0].name == "name1"
    assert np.array_equal(concepts[0].data, np.array([[1, 5], [2, 6]]))
    assert np.array_equal(concepts[0].labels, np.array([0, 0]))
    assert concepts[1].name == "name2"
    assert np.array_equal(concepts[1].data, np.array([[3, 7], [4, 8]]))
    assert np.array_equal(concepts[1].labels, np.array([1, 0]))


def test_read_concepts_fail_if_non_continuous_concepts_id():
    df = sample_df.copy()
    df["concept_id"] = [0, 0, 2, 2]

    with pytest.raises(ValueError):
        read_concepts_from_df(df)


@pytest.mark.parametrize("column", ["concept_id", "concept_name", "label"])
def test_read_concepts_fail_if_missing_columns(column):
    df = sample_df.copy()
    df = df.drop(columns=[column])

    with pytest.raises(ValueError):
        read_concepts_from_df(df)
