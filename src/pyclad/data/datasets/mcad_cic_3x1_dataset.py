from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class McadCic3x1Dataset(TabularCadDataset):
    """
    MCAD-CIC-3x1 benchmark: a multi-source scenario with 3 concepts (``cicids2017``, ``cicids2018``, ``cicunsw``).
    """

    _hf_repo = "lifelonglab/MCAD-CIC-3x1"
    _display_name = "MCAD-CIC-3x1"
