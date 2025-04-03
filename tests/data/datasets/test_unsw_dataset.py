from tempfile import TemporaryDirectory

import pytest

from pyclad.data.datasets.unsw_dataset import UnswDataset


@pytest.mark.longrun
def test_downloading_unsw_dataset():
    with TemporaryDirectory() as tmpdir:
        dataset = UnswDataset(dataset_type="random_anomalies", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10

        dataset = UnswDataset(dataset_type="clustered_with_closest_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10

        dataset = UnswDataset(dataset_type="clustered_with_random_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10
