from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from torch import nn

from pyclad.vision.models.patchcore.config import PatchCoreConfig
from pyclad.vision.models.utilities.backbones import (
    TorchvisionFeatureExtractor,
    default_backbone_return_nodes,
)
from pyclad.vision.models.utilities.base_model import VisionScoringBase


class MeanMapper(nn.Module):
    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class FeaturePreprocessor(nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.input_dims = tuple(input_dims)
        self.output_dim = output_dim
        self.preprocessing_modules = nn.ModuleList([MeanMapper(output_dim) for _ in input_dims])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        reduced = [module(feature) for module, feature in zip(self.preprocessing_modules, features)]
        return torch.stack(reduced, dim=1)


class FeatureAggregator(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class PatchMaker:
    def __init__(self, patchsize: int, stride: int):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)

        number_of_total_patches = []
        for spatial_size in features.shape[-2:]:
            n_patches = (spatial_size + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def unpatch_scores(scores: np.ndarray, batch_size: int) -> np.ndarray:
        return scores.reshape(batch_size, -1, *scores.shape[1:])

    @staticmethod
    def score(x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        while x_t.ndim > 1:
            x_t = torch.max(x_t, dim=-1).values
        return x_t.numpy() if isinstance(x, np.ndarray) else x_t


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
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        random_seed: int = 0,
    ):
        if percentage <= 0.0 or percentage > 1.0:
            raise ValueError("percentage must be in (0, 1]")

        self.percentage = percentage
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

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Greedy coreset selection: repeatedly pick the point with the largest approximate
        min-distance to the points already selected.
        """
        number_of_starting_points = min(self.number_of_starting_points, len(features))
        rng = np.random.default_rng(self.random_seed)
        start_points = rng.choice(len(features), number_of_starting_points, replace=False).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(features, features[start_points])
        approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)

        coreset_indices = []
        num_coreset_samples = max(1, int(len(features) * self.percentage))

        with torch.no_grad():
            for _ in range(num_coreset_samples):
                select_idx = torch.argmax(approximate_coreset_anchor_distances.squeeze(-1)).item()
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
        if self.percentage == 1.0:
            return features

        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        reduced_features = self._reduce_features(feature_tensor)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        return feature_tensor[sample_indices].cpu().numpy().astype(np.float32, copy=False)


class PatchCore(VisionScoringBase):
    """PatchCore: nearest-neighbour scoring against a coreset memory bank of patch features."""

    config: PatchCoreConfig

    def __init__(self, config: Optional[PatchCoreConfig] = None):
        super().__init__(config or PatchCoreConfig())

        self._apply_seed()  # before backbone construction: its init is random when not pretrained
        nodes = self.config.backbone_return_nodes or self._require_nodes(self.config.backbone_name)
        self.module = TorchvisionFeatureExtractor(
            backbone_name=self.config.backbone_name,
            return_nodes=nodes,
            pretrained=self.config.pretrained_backbone,
            freeze=self.config.freeze_backbone,
            weights_name=self.config.pretrained_weights,
        ).to(self._device)
        self.module.eval()

        self._patch_maker = PatchMaker(patchsize=self.config.patchsize, stride=self.config.patchstride)
        feature_dimensions = self.module.infer_out_channels(self.config.input_size)
        self._feature_preprocessor = FeaturePreprocessor(
            input_dims=feature_dimensions,
            output_dim=self.config.pretrain_embed_dimension,
        ).to(self._device)
        self._feature_aggregator = FeatureAggregator(target_dim=self.config.target_embed_dimension).to(self._device)
        self._segmentor = RescaleSegmentor(
            device=self._device,
            target_size=self.config.input_size,
            smoothing=self.config.smoothing_sigma,
        )

        self._memory_bank: Optional[np.ndarray] = None
        self._nn_index: Optional[NearestNeighbors] = None
        self._cached_image_scores: Optional[torch.Tensor] = None

    @staticmethod
    def _require_nodes(backbone_name: str) -> tuple[str, ...]:
        # PatchCore uses the two mid-level residual stages, as in the reference implementation.
        nodes = default_backbone_return_nodes(backbone_name)
        return tuple(node for node in nodes if node in ("layer2", "layer3")) or tuple(nodes[1:3])

    # --- feature extraction --------------------------------------------------
    @staticmethod
    def _align_feature_maps(features: list[torch.Tensor], patch_shapes: list[list[int]]) -> list[torch.Tensor]:
        """Resample every feature level onto the first level's patch grid."""
        reference_shape = patch_shapes[0]
        for index in range(1, len(features)):
            feat = features[index]
            dims = patch_shapes[index]

            feat = feat.reshape(feat.shape[0], dims[0], dims[1], *feat.shape[2:])
            feat = feat.permute(0, -3, -2, -1, 1, 2)
            permuted_shape = feat.shape
            feat = feat.reshape(-1, *feat.shape[-2:])
            feat = F.interpolate(
                feat.unsqueeze(1),
                size=(reference_shape[0], reference_shape[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            feat = feat.reshape(*permuted_shape[:-2], reference_shape[0], reference_shape[1])
            feat = feat.permute(0, -2, -1, 1, 2, 3)
            features[index] = feat.reshape(len(feat), -1, *feat.shape[-3:])

        return features

    def _embed(self, images: torch.Tensor) -> tuple[np.ndarray, list[list[int]]]:
        with torch.no_grad():
            features = self.module(images)
            patches_with_shapes = [self._patch_maker.patchify(f, return_spatial_info=True) for f in features]
            patch_shapes = [shape for _, shape in patches_with_shapes]
            features = [patch for patch, _ in patches_with_shapes]

            features = self._align_feature_maps(features, patch_shapes)
            features = [f.reshape(-1, *f.shape[-3:]) for f in features]
            features = self._feature_aggregator(self._feature_preprocessor(features))

        return features.detach().cpu().numpy().astype(np.float32, copy=False), patch_shapes

    # --- fit -----------------------------------------------------------------
    def fit(self, data: np.ndarray):
        if len(data) == 0:
            return

        self._apply_seed()  # before coreset subsampling: its projection and start points are random

        embeddings = []
        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            batch_embeddings, _ = self._embed(batch_x.to(self._device, dtype=torch.float32))
            embeddings.append(batch_embeddings)

        sampler = ApproximateGreedyCoresetSampler(
            percentage=self.config.coreset_sampling_ratio,
            device=self._device,
            number_of_starting_points=self.config.coreset_starting_points,
            dimension_to_project_features_to=self.config.coreset_projection_dimension,
            random_seed=self.config.seed if self.config.seed is not None else 0,
        )
        self._memory_bank = sampler.run(np.concatenate(embeddings, axis=0))
        self._nn_index = NearestNeighbors(n_neighbors=min(self.config.n_neighbors, len(self._memory_bank)), n_jobs=1)
        self._nn_index.fit(self._memory_bank)

        self._calibrate_threshold(data)

    # --- inference -----------------------------------------------------------
    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        if self._memory_bank is None or self._nn_index is None:
            raise RuntimeError("PatchCore must be fitted before scoring or predicting")

        embeddings, patch_shapes = self._embed(batch)
        distances, _ = self._nn_index.kneighbors(embeddings)
        # sklearn returns plain Euclidean distances; the reference (ADer's FaissNN -> IndexFlatL2)
        # returns SQUARED L2, and squaring isn't a monotone transform after Gaussian smoothing -- so
        # every downstream quantity here uses the reference's squared scale, not sklearn's.
        distances = np.square(distances)
        patch_scores = np.mean(distances, axis=-1).astype(np.float32, copy=False)

        batch_size = batch.shape[0]
        unpatched = self._patch_maker.unpatch_scores(patch_scores, batch_size=batch_size)

        # Image score: max over RAW patch scores, before any smoothing (reference behaviour).
        image_scores = self._patch_maker.score(unpatched.reshape(*unpatched.shape[:2], -1))
        self._cached_image_scores = torch.from_numpy(np.asarray(image_scores, dtype=np.float32).reshape(batch_size)).to(
            batch.device
        )

        scales = patch_shapes[0]
        maps = self._segmentor.convert_to_segmentation(
            unpatched.reshape(batch_size, scales[0], scales[1]).astype(np.float32, copy=False)
        )
        return torch.from_numpy(maps).to(batch.device)

    def _aggregate_scores(self, score_maps: torch.Tensor) -> torch.Tensor:
        cached, self._cached_image_scores = self._cached_image_scores, None
        if cached is not None and cached.shape[0] == score_maps.shape[0]:
            return cached
        return super()._aggregate_scores(score_maps)

    def name(self) -> str:
        return "PatchCore"

    def _extra_info(self) -> dict:
        return {
            "device": str(self._device),
            "feature_layers": list(self.module.return_nodes),
            "memory_bank_size": None if self._memory_bank is None else int(len(self._memory_bank)),
        }
