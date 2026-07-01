from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class CadCicunswDataset(TabularCadDataset):
    """CAD-CICUNSW benchmark (5 concepts). See :class:`TabularCadDataset` for details and the ``ordering`` parameter."""

    _hf_repo = "lifelonglab/CAD-CICUNSW"
    _display_name = "CAD-CICUNSW"
