import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


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
):
    sns.set_theme(style="darkgrid")
    sns.set(rc={"figure.figsize": figsize})

    data = []  # learned_concept, evaluated_concept, metric_value

    for learned_concept in concepts_order:
        for evaluated_concept in concepts_order:
            metric_value = matrix[learned_concept][evaluated_concept]
            data.append(
                [
                    learned_concept if names_mapping is None else names_mapping[learned_concept],
                    evaluated_concept if names_mapping is None else names_mapping[evaluated_concept],
                    metric_value,
                ]
            )

    df = pd.DataFrame(data, columns=["learned_concept", "evaluated_concept", "metric_value"])
    df = df.pivot(index="learned_concept", columns="evaluated_concept", values="metric_value")
    p: Axes = sns.heatmap(
        df, vmin=0, vmax=1, center=0.5, cmap=sns.color_palette(color_palette, as_cmap=True), annot=annotate
    )
    p.set_xlabel(xlabel)
    p.set_ylabel(ylabel)
    p.set_title(title)

    if output_path is not None:
        plt.savefig(output_path)

    return p