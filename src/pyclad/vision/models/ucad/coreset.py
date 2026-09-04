"""UCAD's own copies of two PatchCore utilities.

``RescaleSegmentor`` and ``ApproximateGreedyCoresetSampler`` originate in PatchCore
(``pyclad.vision.models.patchcore.patchcore``), which is not yet on ``main`` -- it lives on
another, still-unmerged feature branch. UCAD needs exactly these two helpers, so this module
carries its own copies (trimmed to only what UCAD calls: ``convert_to_segmentation`` and
``run_with_target_size``) instead of depending on that branch.

When the PatchCore model lands on ``main``, this module should be reduced to imports from
``pyclad.vision.models.patchcore.patchcore`` rather than carrying duplicate implementations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn


class RescaleSegmentor:
    def __init__(self, device: torch.device, target_size: tuple[int, int], smoothing: float):
        self.device = device
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)

            scores = patch_scores.to(self.device).unsqueeze(1)
            scores = F.interpolate(scores, size=self.target_size, mode="bilinear", align_corners=False)
            scores = scores.squeeze(1).cpu().numpy()

        if self.smoothing <= 0:
            return scores.astype(np.float32, copy=False)
        return np.asarray(
            [ndimage.gaussian_filter(score, sigma=self.smoothing) for score in scores],
            dtype=np.float32,
        )


class ApproximateGreedyCoresetSampler:
    """Greedy coreset subsampling, restricted to the exact-count entry point UCAD uses.

    The original PatchCore utility also offers a percentage-based ``run()`` (and the
    ``percentage`` constructor parameter that only it needs) plus a weighted-selection
    variant of the private helper below, for the other model's needs. Neither is reachable
    from UCAD, which only ever asks for an exact-size coreset, so both are left out here.
    """

    def __init__(
        self,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        random_seed: int = 0,
    ):
        self.device = device
        self.number_of_starting_points = number_of_starting_points
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.random_seed = random_seed

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] == self.dimension_to_project_features_to:
            return features.to(self.device)

        with torch.random.fork_rng():
            torch.manual_seed(self.random_seed)
            mapper = nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False).to(self.device)
        return mapper(features.to(self.device))

    @staticmethod
    def _compute_batchwise_differences(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor, num_samples: int) -> np.ndarray:
        """Greedy coreset selection: repeatedly pick the point farthest from the selected set.

        "Farthest" is approximated via mean distance to a random subset of starting points,
        updated incrementally as each new point is selected.
        """
        number_of_starting_points = min(self.number_of_starting_points, len(features))
        rng = np.random.default_rng(self.random_seed)
        start_points = rng.choice(len(features), number_of_starting_points, replace=False).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(features, features[start_points])
        approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)

        coreset_indices = []

        with torch.no_grad():
            for _ in range(num_samples):
                scores = approximate_coreset_anchor_distances.squeeze(-1)
                select_idx = torch.argmax(scores).item()
                coreset_indices.append(select_idx)

                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

    def run_with_target_size(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Select exactly ``target_size`` coreset points, or all of them when there are fewer.

        UCAD specifies its key and knowledge banks as an absolute number of vectors (196);
        expressing that as a percentage of a varying pool size could round down by one, so
        this samples an exact count directly instead.
        """
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        if len(features) <= target_size:
            return features.astype(np.float32, copy=False)

        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        reduced_features = self._reduce_features(feature_tensor)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features, num_samples=target_size)
        return feature_tensor[sample_indices].cpu().numpy().astype(np.float32, copy=False)
