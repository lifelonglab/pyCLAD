from tempfile import TemporaryDirectory

import pytest

from pyclad.data.datasets.cad_cicids2017_dataset import CadCicids2017Dataset
from pyclad.data.datasets.cad_cicids2018_dataset import CadCicids2018Dataset
from pyclad.data.datasets.cad_cicunsw_dataset import CadCicunswDataset
from pyclad.data.datasets.mcad_cic_3x1_dataset import McadCic3x1Dataset
from pyclad.data.datasets.mcad_cic_3xn_dataset import McadCic3xNDataset


@pytest.mark.longrun
@pytest.mark.parametrize(
    "dataset_cls, concepts_no",
    [(CadCicids2017Dataset, 6), (CadCicids2018Dataset, 5), (CadCicunswDataset, 5)],
)
def test_downloading_tabular_cad_dataset(dataset_cls, concepts_no):
    with TemporaryDirectory() as tmpdir:
        dataset = dataset_cls(ordering="curriculum_asc", cache_dir=tmpdir)
        assert len(dataset.train_concepts()) == concepts_no
        assert len(dataset.test_concepts()) == concepts_no

        # Concepts are presented in the order defined by orderings.json; switching to the descending
        # variant must reverse the sequence of concept names.
        asc_order = [c.name for c in dataset.train_concepts()]
        desc = dataset_cls(ordering="curriculum_desc", cache_dir=tmpdir)
        assert [c.name for c in desc.train_concepts()] == list(reversed(asc_order))


@pytest.mark.longrun
@pytest.mark.parametrize(
    "dataset_cls, expected_orderings",
    [
        (CadCicids2017Dataset, {"curriculum_asc", "curriculum_desc", "generalization_asc", "generalization_desc", "smooth_drift", "abrupt_drift"}),
        (CadCicids2018Dataset, {"curriculum_asc", "curriculum_desc", "generalization_asc", "generalization_desc", "smooth_drift", "abrupt_drift"}),
        (CadCicunswDataset, {"curriculum_asc", "curriculum_desc", "generalization_asc", "generalization_desc", "smooth_drift", "abrupt_drift"}),
        (McadCic3x1Dataset, {"curriculum_asc", "curriculum_desc", "generalization_asc", "generalization_desc", "smooth_drift", "abrupt_drift"}),
        (McadCic3xNDataset, {"curriculum_asc", "curriculum_desc", "generalization_asc", "generalization_desc", "smooth_drift", "abrupt_drift"}),
    ],
)
def test_available_orderings(dataset_cls, expected_orderings):
    with TemporaryDirectory() as tmpdir:
        assert expected_orderings.issubset(set(dataset_cls.available_orderings(cache_dir=tmpdir)))


@pytest.mark.longrun
def test_unknown_ordering_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unknown ordering"):
            CadCicids2017Dataset(ordering="does_not_exist", cache_dir=tmpdir)
