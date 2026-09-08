from pyclad.data.datasets.tabular_cad_dataset import TabularCadDataset


class McadCic3x1Dataset(TabularCadDataset):
    """
    MCAD-CIC-3x1 benchmark: a multi-source scenario with 3 concepts (``cicids2017``, ``cicids2018``, ``cicunsw``).

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

    _hf_repo = "lifelonglab/MCAD-CIC-3x1"
    _display_name = "MCAD-CIC-3x1"
