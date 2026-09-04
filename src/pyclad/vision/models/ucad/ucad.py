from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.coreset import (
    ApproximateGreedyCoresetSampler,
    RescaleSegmentor,
)
from pyclad.vision.models.ucad.structure import build_structure_mask_provider
from pyclad.vision.models.utilities.base_model import VisionScoringBase

logger = logging.getLogger(__name__)


def structure_contrastive_loss(
    patch_features: torch.Tensor,
    region_labels: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """UCAD's Structure-based Contrastive Learning objective (paper Eq. 3).

    Patch features assigned to the same SAM region are pulled together and features from
    different regions are pushed apart, over all pairs within an image. This reproduces the
    reference ``VisionTransformer.contrastive_loss`` exactly, including the self-pairs it
    keeps on the diagonal (which contribute a constant ``-1/T`` per patch and therefore do
    not change the gradient direction, but do change the reported loss value).
    """
    if patch_features.ndim != 3:
        raise ValueError(f"patch_features must have shape (B, P, D), got {tuple(patch_features.shape)}")
    if region_labels.shape != patch_features.shape[:2]:
        raise ValueError(
            f"region_labels must have shape {tuple(patch_features.shape[:2])}, " f"got {tuple(region_labels.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    normalized = F.normalize(patch_features, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2)) / temperature
    same_region = (region_labels.unsqueeze(1) == region_labels.unsqueeze(2)).to(similarity.dtype)
    return (-similarity * same_region + torch.exp(similarity) * (1.0 - same_region)).mean()


@dataclass(frozen=True)
class UCADTaskMemory:
    """One entry of UCAD's key-prompt-knowledge memory (paper's CPM)."""

    key: np.ndarray
    prompt: torch.Tensor
    knowledge: np.ndarray
    concept_id: Optional[str] = None
    final_scl_loss: Optional[float] = None


class UCAD(VisionScoringBase):
    """Unsupervised Continual Anomaly Detection with contrastively-learned prompts.

    Each ``fit()`` appends one task memory and never resets, so a plain concept-incremental
    stream builds the whole key-prompt-knowledge space. Inference is task-agnostic: the task
    is chosen per image from the learned keys, never from the concept id.

    No method accepts test data, test labels, or any statistic derived from them. The
    reference implementation selects its prompt and knowledge bank by test AUROC; this port
    trains for a fixed number of epochs and keeps the last one, so its scores are lower than
    the paper's by construction. See ``docs/vision.md``.
    """

    config: UCADConfig

    def __init__(self, config: Optional[UCADConfig] = None):
        super().__init__(config or UCADConfig())

        self._apply_seed()  # before backbone construction: init is random when not pretrained
        self.module = self._build_backbone().to(self._device).eval()
        self._validate_backbone()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

        self._embed_dim = int(self.module.embed_dim)
        self._num_heads = int(self.module.blocks[0].attn.num_heads)
        if self._embed_dim % self._num_heads != 0:
            raise ValueError(
                f"UCAD embedding dimension {self._embed_dim} is not divisible by "
                f"the attention head count {self._num_heads}"
            )
        self._head_dim = self._embed_dim // self._num_heads
        if self.config.prompt_depth > self.config.feature_layer:
            logger.info(
                "UCAD prompt_depth=%d exceeds feature_layer=%d; prompt slices past the read-out "
                "block receive no gradient and only cost storage. This matches the reference, "
                "which prompts all 12 blocks while reading features after block index 5.",
                self.config.prompt_depth,
                self.config.feature_layer,
            )

        self._segmentor = RescaleSegmentor(
            device=self._device,
            target_size=self.config.input_size,
            smoothing=self.config.smoothing_sigma,
        )
        self._mask_provider = build_structure_mask_provider(self.config, str(self._device))
        self._task_memories: list[UCADTaskMemory] = []
        self._key_indices: list[NearestNeighbors] = []
        self._knowledge_indices: list[NearestNeighbors] = []
        self._grid_size: Optional[tuple[int, int]] = None
        self._current_concept_id: Optional[str] = None
        self._cached_image_scores: Optional[torch.Tensor] = None

    # --- construction --------------------------------------------------------
    def _build_backbone(self) -> nn.Module:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("UCAD requires timm. Install pyclad with the 'ucad' extra.") from exc

        try:
            return timm.create_model(
                self.config.backbone_name,
                pretrained=self.config.pretrained_backbone,
                num_classes=0,
                img_size=self.config.input_size,
            )
        except TypeError as exc:
            # Only ViT-family models accept `img_size`; a CNN backbone rejects it here rather
            # than failing later in _validate_backbone with a more confusing message.
            raise TypeError(
                f"UCAD backbone '{self.config.backbone_name}' is not a compatible timm ViT: "
                "it does not accept an img_size argument"
            ) from exc

    def _validate_backbone(self) -> None:
        required = ("patch_embed", "_pos_embed", "patch_drop", "norm_pre", "blocks", "embed_dim")
        missing = [name for name in required if not hasattr(self.module, name)]
        if missing:
            raise TypeError(
                f"UCAD backbone '{self.config.backbone_name}' is not a compatible timm ViT; missing {missing}"
            )
        if self.config.feature_layer > len(self.module.blocks):
            raise ValueError(
                f"feature_layer={self.config.feature_layer} exceeds backbone depth={len(self.module.blocks)}"
            )
        if self.config.prompt_depth > len(self.module.blocks):
            raise ValueError(
                f"prompt_depth={self.config.prompt_depth} exceeds backbone depth={len(self.module.blocks)}"
            )

    def _initial_prompt(self) -> torch.Tensor:
        shape = (self.config.prompt_depth, 2, self.config.prompt_length, self._num_heads, self._head_dim)
        if self.config.prompt_init == "zero":
            return torch.zeros(shape, dtype=torch.float32, device=self._device)
        # uniform(-1, 1), matching EPrompt's `nn.init.uniform_(self.prompt, -1, 1)`. Drawn from
        # the global RNG, which _apply_seed() has just fixed, so tasks start from the same
        # initialisation the reference gives them (it re-seeds per task with the same seed).
        return torch.empty(shape, dtype=torch.float32, device=self._device).uniform_(-1.0, 1.0)

    def _forward_block_with_prefix_prompt(
        self, block: nn.Module, tokens: torch.Tensor, layer_prompt: torch.Tensor
    ) -> torch.Tensor:
        """Run one timm ViT block with learned key/value prefixes (DualPrompt-style)."""
        attention = block.attn
        normalized = block.norm1(tokens)
        batch_size, token_count, embed_dim = normalized.shape

        qkv = (
            attention.qkv(normalized)
            .reshape(batch_size, token_count, 3, attention.num_heads, embed_dim // attention.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query, key = attention.q_norm(query), attention.k_norm(key)

        key_prefix = layer_prompt[0].permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        value_prefix = layer_prompt[1].permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        key = torch.cat([key_prefix.to(key.dtype), key], dim=2)
        value = torch.cat([value_prefix.to(value.dtype), value], dim=2)

        weights = attention.attn_drop(torch.matmul(query * attention.scale, key.transpose(-2, -1)).softmax(dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        attended = attention.proj_drop(attention.proj(attended))

        tokens = tokens + block.drop_path1(block.ls1(attended))
        return tokens + block.drop_path2(block.ls2(block.mlp(block.norm2(tokens))))

    def _forward_patch_tokens(self, images: torch.Tensor, prompt: Optional[torch.Tensor]) -> torch.Tensor:
        """Patch tokens after ``feature_layer`` blocks, prefix tokens dropped."""
        x = self.module.patch_embed(images)
        x = self.module._pos_embed(x)
        x = self.module.patch_drop(x)
        x = self.module.norm_pre(x)

        for block_index, block in enumerate(self.module.blocks[: self.config.feature_layer]):
            if prompt is not None and block_index < self.config.prompt_depth:
                x = self._forward_block_with_prefix_prompt(block, x, prompt[block_index])
            else:
                x = block(x)
        return x[:, int(getattr(self.module, "num_prefix_tokens", 1)) :]

    # --- features ------------------------------------------------------------
    def _map_dimension(self, features: torch.Tensor) -> torch.Tensor:
        """768 -> target_embed_dimension, matching the reference MeanMapper + Aggregator."""
        if features.shape[-1] == self.config.target_embed_dimension:
            return features
        flattened = features.reshape(-1, 1, features.shape[-1])
        mapped = F.adaptive_avg_pool1d(flattened, self.config.target_embed_dimension)
        return mapped.reshape(*features.shape[:-1], self.config.target_embed_dimension)

    @staticmethod
    def _infer_grid_size(patch_count: int) -> tuple[int, int]:
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError(f"UCAD requires a square ViT patch grid, got {patch_count} patch tokens")
        return side, side

    def _patch_features(
        self, images: torch.Tensor, prompt: Optional[torch.Tensor]
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Mapped patch embeddings ``(B*P, D)`` for a preprocessed batch, plus the patch grid."""
        self.module.eval()
        with torch.no_grad():
            tokens = self._forward_patch_tokens(
                images.to(self._device, dtype=torch.float32),
                None if prompt is None else prompt.to(self._device),
            )
            grid = self._infer_grid_size(tokens.shape[1])
            mapped = self._map_dimension(tokens)
        return mapped.reshape(-1, mapped.shape[-1]).cpu().numpy().astype(np.float32, copy=False), grid

    def _extract_patch_embeddings(
        self, data: np.ndarray, prompt: Optional[torch.Tensor]
    ) -> tuple[np.ndarray, tuple[int, int]]:
        batches: list[np.ndarray] = []
        grid: Optional[tuple[int, int]] = None
        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            features, grid = self._patch_features(batch_x, prompt)
            batches.append(features)
        return np.concatenate(batches, axis=0), grid

    @property
    def task_memories(self) -> tuple[UCADTaskMemory, ...]:
        return tuple(self._task_memories)

    def set_current_concept(self, concept_id: Optional[str]) -> None:
        """Record which concept the next ``fit()`` learns.

        Used only to locate that concept's structure masks. It never reaches inference:
        ``_inference_maps`` routes from the learned keys alone.
        """
        self._current_concept_id = concept_id

    def _select_coreset(self, features: np.ndarray, target_size: int) -> np.ndarray:
        sampler = ApproximateGreedyCoresetSampler(
            device=self._device,
            number_of_starting_points=self.config.coreset_starting_points,
            dimension_to_project_features_to=self.config.coreset_projection_dimension,
            random_seed=self.config.seed if self.config.seed is not None else 0,
        )
        return sampler.run_with_target_size(features, target_size)

    def _downsample_region_labels(self, masks: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Region-id maps -> one integer label per patch.

        The reference uses ``cv2.resize`` (bilinear, then uint8 rounding) to reach the 14x14
        grid, so the default reproduces that including the rounding.
        """
        if self.config.mask_interpolation == "nearest":
            resized = F.interpolate(masks, size=grid, mode="nearest")
        elif self.config.mask_interpolation == "bilinear":
            resized = F.interpolate(masks, size=grid, mode="bilinear", align_corners=False).round()
        else:
            raise ValueError(f"Unknown UCAD mask interpolation: {self.config.mask_interpolation!r}")
        return resized[:, 0].reshape(masks.shape[0], -1).to(torch.long)

    def _train_prompt(
        self, data: np.ndarray, structure_masks: Optional[np.ndarray]
    ) -> tuple[torch.Tensor, Optional[float]]:
        initial_prompt = self._initial_prompt()
        if structure_masks is None or self.config.epochs == 0:
            return initial_prompt.detach(), None
        if len(structure_masks) != len(data):
            raise ValueError(f"UCAD received {len(structure_masks)} structure masks for {len(data)} images")

        mask_array = np.asarray(structure_masks)
        expected_spatial_size = self._preprocessor.spatial_size(data)
        if tuple(mask_array.shape[1:3]) != expected_spatial_size:
            raise ValueError(
                f"UCAD structure masks have spatial shape {tuple(mask_array.shape[1:3])} but the "
                f"input images have spatial shape {expected_spatial_size}. Structure mask "
                "providers assume NHWC images; check config.input_layout if the images are NCHW."
            )

        images = self._preprocessor.transform(data)
        masks = torch.as_tensor(mask_array, dtype=torch.float32).unsqueeze(1)

        prompt = nn.Parameter(initial_prompt.detach().clone())
        optimizer = torch.optim.Adam(
            [prompt],
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.config.seed) if self.config.seed is not None else None
        loader = DataLoader(
            TensorDataset(images, masks),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        self.module.eval()
        final_loss: Optional[float] = None
        for epoch in range(self.config.epochs):
            epoch_losses: list[float] = []
            for image_batch, mask_batch in loader:
                image_batch = image_batch.to(self._device, dtype=torch.float32)
                mask_batch = mask_batch.to(self._device)

                tokens = self._forward_patch_tokens(image_batch, prompt)
                labels = self._downsample_region_labels(mask_batch, self._infer_grid_size(tokens.shape[1]))
                loss = structure_contrastive_loss(tokens, labels, temperature=self.config.contrastive_temperature)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([prompt], self.config.gradient_clip_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            final_loss = float(np.mean(epoch_losses)) if epoch_losses else final_loss
            if self.config.show_training_progress:
                logger.info(
                    "UCAD task %d SCL epoch %d/%d loss=%s",
                    len(self._task_memories),
                    epoch + 1,
                    self.config.epochs,
                    "nan" if final_loss is None else f"{final_loss:.6f}",
                )
        return prompt.detach(), final_loss

    def fit(self, data: np.ndarray) -> None:
        if len(data) == 0:
            raise ValueError("UCAD cannot learn an empty task")
        self._apply_seed()

        key_features, grid = self._extract_patch_embeddings(data, prompt=None)
        if self._grid_size is None:
            self._grid_size = grid
        elif self._grid_size != grid:
            raise ValueError(f"All UCAD tasks must share the same feature grid; expected {self._grid_size}, got {grid}")
        key = self._select_coreset(key_features, self.config.key_size)

        structure_masks = self._mask_provider.masks_for(self._current_concept_id, np.asarray(data))
        if structure_masks is None and self.config.epochs > 0:
            logger.info(
                "UCAD task %d runs the CPM-only ablation (structure_mode='none'): no SCL performed.",
                len(self._task_memories),
            )
        prompt, final_scl_loss = self._train_prompt(data, structure_masks)
        self._last_loss = final_scl_loss

        knowledge_features, _ = self._extract_patch_embeddings(data, prompt=prompt)
        knowledge = self._select_coreset(knowledge_features, self.config.knowledge_size)

        self._task_memories.append(
            UCADTaskMemory(
                key=key,
                prompt=prompt.detach().cpu(),
                knowledge=knowledge,
                concept_id=self._current_concept_id,
                final_scl_loss=final_scl_loss,
            )
        )
        self._key_indices.append(NearestNeighbors(n_neighbors=1, n_jobs=1).fit(key))
        self._knowledge_indices.append(
            NearestNeighbors(n_neighbors=min(self.config.n_neighbors, len(knowledge)), n_jobs=1).fit(knowledge)
        )
        self._calibrate_threshold(data)

    def _require_fitted(self) -> None:
        if not self._task_memories or self._grid_size is None:
            raise RuntimeError("UCAD must learn at least one task before scoring or predicting")

    def _route(self, batch: torch.Tensor) -> np.ndarray:
        """Task index per image, from the learned keys alone (paper Eq. 4).

        The concept id of the evaluated data is never consulted, which is what makes the
        evaluation task-agnostic. Distances are squared to match the reference's faiss index;
        this matters here because the statistic is a mean, which squaring does not commute with.
        """
        batch_size = batch.shape[0]
        if len(self._task_memories) == 1:
            return np.zeros(batch_size, dtype=np.int64)

        features, grid = self._patch_features(batch, prompt=None)
        queries = features.reshape(batch_size, grid[0] * grid[1], -1)

        statistics = np.empty((len(self._task_memories), batch_size), dtype=np.float32)
        for task_index, index in enumerate(self._key_indices):
            for image_index, query in enumerate(queries):
                distances, _ = index.kneighbors(query)
                squared = np.square(distances[:, 0])
                statistics[task_index, image_index] = (
                    squared.max() if self.config.routing_statistic == "max" else squared.mean()
                )
        return np.argmin(statistics, axis=0).astype(np.int64)

    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        self._require_fitted()

        batch_size = batch.shape[0]
        grid_height, grid_width = self._grid_size
        patch_scores = np.empty((batch_size, grid_height * grid_width), dtype=np.float32)

        selected = self._route(batch)
        for task_index in np.unique(selected):
            positions = np.flatnonzero(selected == task_index)
            memory = self._task_memories[int(task_index)]
            features, _ = self._patch_features(batch[positions], memory.prompt)
            distances, _ = self._knowledge_indices[int(task_index)].kneighbors(features)
            scores = np.square(distances).mean(axis=1).astype(np.float32, copy=False)
            patch_scores[positions] = scores.reshape(len(positions), -1)

        self._cached_image_scores = torch.from_numpy(patch_scores.max(axis=1)).to(batch.device)
        maps = self._segmentor.convert_to_segmentation(patch_scores.reshape(batch_size, grid_height, grid_width))
        return torch.from_numpy(np.asarray(maps, dtype=np.float32)).to(batch.device)

    def _aggregate_scores(self, score_maps: torch.Tensor) -> torch.Tensor:
        cached, self._cached_image_scores = self._cached_image_scores, None
        if cached is not None and cached.shape[0] == score_maps.shape[0]:
            return cached
        return super()._aggregate_scores(score_maps)

    def selected_task_indices(self, data: np.ndarray) -> np.ndarray:
        """Task chosen per image, for routing-accuracy diagnostics."""
        self._require_fitted()
        selected: list[np.ndarray] = []
        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            selected.append(self._route(batch_x.to(self._device, dtype=torch.float32)))
        return np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)

    def _variant_label(self) -> str:
        """Report what actually ran, not what was configured.

        ``config.structure_mode`` is intent: ``_train_prompt`` silently falls back to an
        untrained prompt (no SCL) whenever structure masks are unavailable for a concept, or
        whenever ``config.epochs == 0``, regardless of ``structure_mode``. Since this label
        lands in benchmark result JSON, it must be derived from what happened, not from
        config. ``UCADTaskMemory.final_scl_loss`` is ``None`` exactly when no SCL ran for that
        task, so it is the honest signal to key off.
        """
        if not self._task_memories:
            return "no tasks fitted yet"
        trained = [memory.final_scl_loss is not None for memory in self._task_memories]
        if all(trained):
            return "CPM+SCL"
        if not any(trained):
            return "CPM-only"
        return "mixed CPM-only/CPM+SCL"

    def _extra_info(self) -> dict:
        memory_bytes = sum(
            memory.key.nbytes + memory.knowledge.nbytes + memory.prompt.numel() * memory.prompt.element_size()
            for memory in self._task_memories
        )
        return {
            "method": "UCAD",
            "paper": "Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt (AAAI 2024)",
            "variant": self._variant_label(),
            "device": str(self._device),
            "tasks_seen": len(self._task_memories),
            "task_concepts": [memory.concept_id for memory in self._task_memories],
            "key_sizes": [int(len(memory.key)) for memory in self._task_memories],
            "knowledge_sizes": [int(len(memory.knowledge)) for memory in self._task_memories],
            "prompt_shapes": [list(memory.prompt.shape) for memory in self._task_memories],
            "final_scl_losses": [memory.final_scl_loss for memory in self._task_memories],
            "key_prompt_knowledge_bytes": int(memory_bytes),
            "epoch_selection": "last",
        }

    def name(self) -> str:
        return "UCAD"
