import numpy as np

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrixMetric, ConceptLevelMatrix


# from evaluation.metrics.concepts_metric import ConceptsMatrixMetric, MetricMatrix
#
#
# class ContinualAverageAcrossLearnedConcepts(ConceptsMatrixMetric):
#
#     def compute(self, metric_matrix: MetricMatrix) -> float:
#         concepts_no = len(metric_matrix)
#         values = []
#
#         for learned_task in range(concepts_no):
#             for evaluated_task in range(learned_task + 1):
#                 values.append(metric_matrix[learned_task][evaluated_task])
#
#         return np.mean(values)
#
#     def name(self) -> str:
#         return 'ContinualAverageAcrossLearnedConcepts'


class ContinualAverageAcrossLearnedConcepts(ConceptLevelMatrixMetric):

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        concepts_no = len(metric_matrix)
        values = []

        for learned_task in range(concepts_no):
            for evaluated_task in range(learned_task + 1):
                values.append(metric_matrix[learned_task][evaluated_task])

        return np.mean(values)

    def name(self) -> str:
        return "ContinualAverageAcrossLearnedConcepts"
