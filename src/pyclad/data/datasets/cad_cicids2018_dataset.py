from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class CadCicids2018Dataset(TabularCadDataset):
    """CAD-CICIDS2018 benchmark (5 concepts). See :class:`TabularCadDataset` for details and the ``ordering`` parameter."""

    _hf_repo = "lifelonglab/CAD-CICIDS2018"
    _display_name = "CAD-CICIDS2018"
