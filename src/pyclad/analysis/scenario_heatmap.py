import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _create_upper_diagonal_mask(tasks_no: int) -> np.array:
    mask = np.zeros((tasks_no, tasks_no))
    for i in range(tasks_no):
        for j in range(tasks_no):
            if j > i:
                mask[i, j] = True
    return mask


def plot_metric_heatmap(
    matrix: Dict,
    concepts_order: List[str],
    output_path: pathlib.Path = None,
    names_mapping: Dict[str, str] = None,
    xlabel: str = "Evaluating on concept",
    ylabel: str = "After learning concept",
    title: str = "Performance Heatmap",
    annotate: bool = False,
    color_palette: str = "plasma",
    figsize: tuple = (6, 5),
    ignore_upper_diagonal: bool = False,
):
    matrix_keys = set(matrix.keys())
    assert all(
        set(concept_results.keys()) == matrix_keys for concept_results in matrix.values()
    ), "The outer dict and every inner dict must have the same keys."  # TODO: Allow for different keys in the inner dicts

    sns.set_theme(style="darkgrid", rc={"figure.figsize": figsize})

    df = pd.DataFrame(
        [[matrix[learned][evaluated] for evaluated in concepts_order] for learned in concepts_order],
        index=concepts_order,
        columns=concepts_order,
    )
    if names_mapping is not None:
        df = df.rename(index=names_mapping, columns=names_mapping)

    p: Axes = sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        center=0.5,
        cmap=sns.color_palette(color_palette, as_cmap=True),
        annot=annotate,
        mask=_create_upper_diagonal_mask(len(concepts_order)) if ignore_upper_diagonal else None,
    )
    p.set_xlabel(xlabel)
    p.set_ylabel(ylabel)
    p.set_title(title)

    if output_path is not None:
        plt.tight_layout()
        plt.savefig(output_path)

    return p
