import json
import pathlib

from pyclad.analysis.scenario_heatmap import plot_metric_heatmap

if __name__ == "__main__":
    results_path = pathlib.Path("output.json")  # you need to generate this file using concept_aware_examply.py
    with open(results_path) as fp:
        loaded_data = json.load(fp)
        concepts_order = loaded_data["conceptMetricCallback_ROC-AUC"]["ROC-AUC"]["conceptsOrder"]
        metric_matrix = loaded_data["conceptMetricCallback_ROC-AUC"]["ROC-AUC"]["metricMatrix"]
        names_mapping = {"Cluster_0": "C0", "Cluster_1": "C1", "Cluster_2": "C2", "Cluster_3": "C3", "Cluster_4": "C4"}
        plot_metric_heatmap(
            metric_matrix, concepts_order, names_mapping=names_mapping, output_path=pathlib.Path("heatmap.pdf")
        )