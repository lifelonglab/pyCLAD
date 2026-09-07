"""Coreset sampling and anomaly-map rescaling, shared by the patch-memory models.

These two helpers originate in PatchCore and are needed verbatim by every model that scores
patches against a subsampled memory bank. They live here, in ``utilities``, rather than inside
one model's package so that no model has to import from another -- the same layering rule that
keeps models from importing strategy packages.

Two sizings are offered because the callers ask for different things: PatchCore keeps a
fraction of its feature pool (``run``), UCAD keeps an exact count (``run_with_target_size``).
Both go through the same greedy selection.
"""

from __future__ import annotations

from typing import Optional

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
    """Greedy coreset subsampling: repeatedly keep the point farthest from what is already kept.

    ``percentage`` is only read by ``run``; a caller that sizes its coreset absolutely uses
    ``run_with_target_size`` and leaves it unset.
    """

    def __init__(
        self,
        device: torch.device,
        percentage: Optional[float] = None,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        random_seed: int = 0,
    ):
        if percentage is not None and not 0.0 < percentage <= 1.0:
            raise ValueError("percentage must be in (0, 1]")

        self.device = device
        self.percentage = percentage
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

    def run(self, features: np.ndarray) -> np.ndarray:
        """Keep ``percentage`` of ``features``, at least one point."""
        if self.percentage is None:
            raise ValueError("run() needs a percentage; pass one to the constructor or call run_with_target_size()")
        if self.percentage == 1.0:
            return features

        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        reduced_features = self._reduce_features(feature_tensor)
        num_samples = max(1, int(len(features) * self.percentage))
        sample_indices = self._compute_greedy_coreset_indices(reduced_features, num_samples=num_samples)
        return feature_tensor[sample_indices].cpu().numpy().astype(np.float32, copy=False)

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
