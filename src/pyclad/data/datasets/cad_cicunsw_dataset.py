from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class CadCicunswDataset(TabularCadDataset):
    """
    CAD-CICUNSW benchmark (5 concepts). See :class:`TabularCadDataset` for details and the ``ordering`` parameter.

    If using, please cite:

    .. code-block:: bibtex

        @misc{faber2026principledcontinualanomalydetection,
              title={Towards Principled Continual Anomaly Detection: A Systematic Framework and Benchmark Scenarios},
              author={Kamil Faber and Mateusz Smendowski and Roberto Corizzo},
              year={2026},
              eprint={2607.18289},
              archivePrefix={arXiv},
              primaryClass={cs.LG},
              url={https://arxiv.org/abs/2607.18289},
        }
    """

    _hf_repo = "lifelonglab/CAD-CICUNSW"
    _display_name = "CAD-CICUNSW"
