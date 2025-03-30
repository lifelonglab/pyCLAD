from tempfile import TemporaryDirectory

import pytest

from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset


@pytest.mark.longrun
def test_downloading_energy_plants_dataset():
    with TemporaryDirectory() as tmpdir:
        dataset = EnergyPlantsDataset(dataset_type="random_anomalies", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10

        dataset = EnergyPlantsDataset(dataset_type="clustered_with_closest_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10

        dataset = EnergyPlantsDataset(dataset_type="clustered_with_random_assignment", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == 10
        assert len(dataset.test_concepts()) == 10
