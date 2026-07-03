from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class McadCic3xNDataset(TabularCadDataset):
    """
    MCAD-CIC-3xN benchmark: a multi-source scenario with 13 concepts drawn from CIC-IDS2017, CIC-IDS2018, and
    CIC-UNSW. See :class:`TabularCadDataset` for details and the ``ordering`` parameter.
    """

    _hf_repo = "lifelonglab/MCAD-CIC-3xN"
    _display_name = "MCAD-CIC-3xN"
