from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.vision.models.fastflow.backbones import (
    TorchvisionFeatureExtractor,
    default_fastflow_return_nodes,
)
from pyclad.vision.models.fastflow.flow import FastFlowSequence


class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size: tuple[int, int]):
        super().__init__()
        self.input_size = tuple(input_size)

    def forward(self, hidden_variables: Sequence[torch.Tensor]) -> torch.Tensor:
        flow_maps: list[torch.Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -0.5 * torch.mean(hidden_variable**2, dim=1, keepdim=True)
            probability = torch.exp(log_prob)
            flow_map = F.interpolate(
                -probability,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)

        return torch.mean(torch.stack(flow_maps, dim=-1), dim=-1)


class FastFlowArchitecture(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        backbone_name: str,
        backbone_return_nodes: Optional[tuple[str, ...]],
        pretrained_backbone: bool,
        freeze_backbone: bool,
        normalize_features: bool,
        flow_steps: int,
        conv3x3_only: bool,
        hidden_ratio: float,
        affine_clamping: float,
    ):
        super().__init__()

        if backbone_return_nodes is None:
            backbone_return_nodes = default_fastflow_return_nodes(backbone_name)

        self.input_size = tuple(input_size)
        self.freeze_backbone = freeze_backbone
        self.return_nodes = tuple(backbone_return_nodes)
        self.feature_extractor = TorchvisionFeatureExtractor(
            backbone_name=backbone_name,
            return_nodes=self.return_nodes,
            pretrained=pretrained_backbone,
            freeze=freeze_backbone,
        )

        self.feature_shapes = tuple(self._infer_feature_shapes(self.input_size))

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(list(shape), elementwise_affine=True) if normalize_features else nn.Identity()
                for shape in self.feature_shapes
            ]
        )
        self.fast_flow_blocks = nn.ModuleList(
            [
                FastFlowSequence(
                    channels=shape[0],
                    flow_steps=flow_steps,
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    affine_clamping=affine_clamping,
                )
                for shape in self.feature_shapes
            ]
        )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=self.input_size)

    def _infer_feature_shapes(self, input_size: tuple[int, int]) -> list[tuple[int, int, int]]:
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            device = next(self.feature_extractor.parameters()).device
            dummy = torch.zeros((1, 3, input_size[0], input_size[1]), dtype=torch.float32, device=device)
            features = self.feature_extractor(dummy)
        self.feature_extractor.train(was_training)

        return [tuple(int(dimension) for dimension in feature.shape[1:]) for feature in features]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self

    def _extract_features(self, batch: torch.Tensor) -> list[torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.feature_extractor(batch)
        else:
            features = self.feature_extractor(batch)
        return [norm(feature) for norm, feature in zip(self.norms, features)]

    def forward(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = self._extract_features(batch)

        hidden_variables: list[torch.Tensor] = []
        log_jacobians: list[torch.Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)
        return hidden_variables, log_jacobians

    def inference(self, batch: torch.Tensor) -> torch.Tensor:
        hidden_variables, _ = self.forward(batch)
        return self.anomaly_map_generator(hidden_variables)[:, 0]

    @staticmethod
    def fastflow_loss(hidden_variables: Sequence[torch.Tensor], log_jacobians: Sequence[torch.Tensor]) -> torch.Tensor:
        loss = hidden_variables[0].new_tensor(0.0)
        for hidden_variable, log_jacobian in zip(hidden_variables, log_jacobians):
            loss = loss + torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - log_jacobian)
        return loss
