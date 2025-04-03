import json
import pathlib

from pyclad.analysis.scenario_heatmap import plot_metric_heatmap

if __name__ == "__main__":
    """This example showcase how to generate ROC-AUC heatmap for the results of concept aware scenario"""
    results_path = pathlib.Path("output.json")  # you need to generate this file using concept_aware_examply.py
    with open(results_path) as fp:
        loaded_data = json.load(fp)
        concepts_order = loaded_data["concept_metric_callback_ROC-AUC"]["concepts_order"]
        metric_matrix = loaded_data["concept_metric_callback_ROC-AUC"]["metric_matrix"]
        # names_mapping = {"Cluster1": "C0", "Concept1": "C1", "Cluster_2": "C2", "Cluster_3": "C3", "Cluster_4": "C4"}
        names_mapping = {"concept1": "C1", "concept2": "C2", "concept3": "C3", "concept4": "C4"}
        plot_metric_heatmap(
            metric_matrix,
            concepts_order,
            names_mapping=names_mapping,
            annotate=True,
            output_path=pathlib.Path("heatmap.pdf"),
            ignore_upper_diagonal=True
        )
