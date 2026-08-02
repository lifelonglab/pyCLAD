from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pyclad.analysis import scenario_heatmap


def test_plot_metric_heatmap_uses_mapped_concept_names_for_display():
    matrix = {
        "concept1": {"concept1": 1.0, "concept2": 0.2},
        "concept2": {"concept1": 0.8, "concept2": 0.9},
    }
    names_mapping = {"concept1": "C1", "concept2": "C2"}

    axes = scenario_heatmap.plot_metric_heatmap(matrix, ["concept1", "concept2"], names_mapping=names_mapping)

    expected_values = [[1.0, 0.2], [0.8, 0.9]]
    expected_labels = ["C1", "C2"]
    try:
        assert isinstance(axes, Axes)
        assert [label.get_text() for label in axes.get_xticklabels()] == expected_labels
        assert [label.get_text() for label in axes.get_yticklabels()] == expected_labels
        assert axes.collections[0].get_array().reshape(2, 2).tolist() == expected_values
    finally:
        plt.close(axes.figure)
