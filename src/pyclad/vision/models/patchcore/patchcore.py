from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn

from pyclad.vision.models.patchcore.config import PatchCoreConfig
from pyclad.vision.models.utilities.backbones import (
    TorchvisionFeatureExtractor,
    default_backbone_return_nodes,
)
from pyclad.vision.models.utilities.base_model import VisionScoringBase
from pyclad.vision.models.utilities.coreset import (
    ApproximateGreedyCoresetSampler,
    RescaleSegmentor,
)


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
