from tempfile import TemporaryDirectory

import pytest

from src.pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset


@pytest.mark.longrun
def test_downloading_nsl_kdd():
    with TemporaryDirectory() as tmpdir:
        dataset = NslKddDataset(dataset_type="random_anomalies", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 20
        assert len(dataset.test_concepts()) == 20

        dataset = NslKddDataset(dataset_type="clustered_with_closest_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 20
        assert len(dataset.test_concepts()) == 20

        dataset = NslKddDataset(dataset_type="clustered_with_random_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 20
        assert len(dataset.test_concepts()) == 20
