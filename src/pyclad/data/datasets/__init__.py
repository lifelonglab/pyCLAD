"""The generic :class:`ConceptsDataset` and the built-in benchmarks, downloaded from Hugging Face on first use."""

from pyclad.data.datasets.cad_cicids2017_dataset import CadCicids2017Dataset
from pyclad.data.datasets.cad_cicids2018_dataset import CadCicids2018Dataset
from pyclad.data.datasets.cad_cicunsw_dataset import CadCicunswDataset
from pyclad.data.datasets.cad_miniboone_dataset import CadMiniBooNeDataset
from pyclad.data.datasets.cad_scania_dataset import CadScaniaDataset
from pyclad.data.datasets.cad_tcm_dataset import CadTcmDataset
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset
from pyclad.data.datasets.mcad_cic_3x1_dataset import McadCic3x1Dataset
from pyclad.data.datasets.mcad_cic_3xn_dataset import McadCic3xNDataset
from pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset
from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset
from pyclad.data.datasets.unsw_dataset import UnswDataset
from pyclad.data.datasets.wind_energy_dataset import WindEnergyDataset

__all__ = [
    "CadCicids2017Dataset",
    "CadCicids2018Dataset",
    "CadCicunswDataset",
    "CadMiniBooNeDataset",
    "CadScaniaDataset",
    "CadTcmDataset",
    "ConceptsDataset",
    "EnergyPlantsDataset",
    "McadCic3x1Dataset",
    "McadCic3xNDataset",
    "NslKddDataset",
    "TabularCadDataset",
    "UnswDataset",
    "WindEnergyDataset",
]
