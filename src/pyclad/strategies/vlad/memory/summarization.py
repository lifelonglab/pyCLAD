from typing import Dict

import numpy as np
from sklearn.cluster import KMeans


def pyramid_budget(layer_by_concept: Dict[str, int], memory_bound: int) -> Dict[str, int]:
    """Per-concept sample budget for Pyramidal Summarization (VLAD paper, Eq. 5).

    Allocates ``memory_bound`` across concepts with a power-of-two decay by hierarchy depth: a
    concept at layer ``l`` (root = 1) gets a share proportional to ``1 / 2^(l-1)``, so root
    concepts are represented more broadly than their sub-concepts. Every concept gets at least 1
    sample, so summarization never erases one outright.

    :param layer_by_concept: concept id -> hierarchy layer (1 = root, 2 = direct sub-concept, ...).
    :param memory_bound: total sample budget across all concepts (``M_B`` in the paper).
    """
    if not layer_by_concept:
        return {}
    weights = {concept_id: 1.0 / (2 ** (layer - 1)) for concept_id, layer in layer_by_concept.items()}
    total_weight = sum(weights.values())
    return {concept_id: max(1, round(memory_bound * weight / total_weight)) for concept_id, weight in weights.items()}


def kmeans_summarize(samples: np.ndarray, budget: int, n_centroids: int, rng: np.random.Generator) -> np.ndarray:
    """Within-concept summarization via k-means coreset selection (VLAD paper, Eq. 6-8).

    Clusters ``samples`` into ``n_centroids`` groups, then keeps the ``budget // n_centroids``
    points closest to each centroid, dropping boundary/outlier points. Returned unchanged if
    already within ``budget``.

    :param samples: array of shape (n, n_features) to summarize.
    :param budget: total number of samples to retain for this concept.
    :param n_centroids: number of k-means clusters (``M_k`` in the paper).
    """
    n = len(samples)
    if n <= budget:
        return samples
    if budget <= 0:
        return samples[:0]

    k = max(1, min(n_centroids, n))
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(samples)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    per_cluster_budget = max(1, budget // k)
    selected = []
    for cluster_id in range(k):
        cluster_samples = samples[labels == cluster_id]
        if len(cluster_samples) == 0:
            continue
        distances_to_centroid = np.linalg.norm(cluster_samples - centers[cluster_id], axis=1)
        closest_order = np.argsort(distances_to_centroid)
        selected.append(cluster_samples[closest_order[:per_cluster_budget]])

    summarized = np.concatenate(selected) if selected else samples[:budget]
    return summarized[:budget]
