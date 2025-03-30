from tempfile import TemporaryDirectory

import pytest

from pyclad.data.datasets.wind_energy_dataset import WindEnergyDataset


@pytest.mark.longrun
def test_downloading_wind_energy_dataset():
    with TemporaryDirectory() as tmpdir:
        dataset = WindEnergyDataset(dataset_type="random_anomalies", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 5
        assert len(dataset.test_concepts()) == 5

        dataset = WindEnergyDataset(dataset_type="clustered_with_closest_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10

        dataset = WindEnergyDataset(dataset_type="clustered_with_random_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 5
        assert len(dataset.test_concepts()) == 5
