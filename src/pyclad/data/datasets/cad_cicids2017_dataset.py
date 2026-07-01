from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class CadCicids2017Dataset(TabularCadDataset):
    """CAD-CICIDS2017 benchmark (6 concepts). See :class:`TabularCadDataset` for details and the ``ordering`` parameter."""

    _hf_repo = "lifelonglab/CAD-CICIDS2017"
    _display_name = "CAD-CICIDS2017"
