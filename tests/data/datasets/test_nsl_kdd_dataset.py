from tempfile import TemporaryDirectory

import pytest

from src.pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset


@pytest.mark.longrun
def test_downloading_nsl_kdd():
    with TemporaryDirectory() as tmpdir:
        dataset = NslKddDataset(dataset_type="random_anomalies", cache_dir=tmpdir)
        assert dataset.name() == "NSL-KDD-random_anomalies"

        dataset = NslKddDataset(dataset_type="clustered_with_closest_assignment", cache_dir=tmpdir)
        assert dataset.name() == "NSL-KDD-clustered_with_closest_assignment"

        dataset = NslKddDataset(dataset_type="clustered_with_random_assignment", cache_dir=tmpdir)
        assert dataset.name() == "NSL-KDD-clustered_with_random_assignment"
