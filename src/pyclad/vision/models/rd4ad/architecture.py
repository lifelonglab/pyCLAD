from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from pyclad.vision.models.rd4ad.standard import (
    build_decoder,
    build_encoder_and_bn,
    supported_backbone_names,
)


class RD4ADArchitecture(nn.Module):
    """Reverse Distillation network: frozen teacher encoder, OCBE bottleneck, student decoder."""

    _SMOOTHING_TRUNCATE = 4.0

    def __init__(
        self,
        backbone_name: str,
        input_size: tuple[int, int],
        pretrained_encoder: bool,
        freeze_encoder: bool,
        score_smoothing_sigma: float,
    ):
        super().__init__()

        if backbone_name not in supported_backbone_names():
            raise ValueError(
                f"Unsupported RD4AD backbone '{backbone_name}'. "
                f"Supported backbones: {', '.join(supported_backbone_names())}"
            )

        self.input_size = tuple(input_size)
        self.freeze_encoder = freeze_encoder
        self.score_smoothing_sigma = score_smoothing_sigma

        self.encoder, self.bn = build_encoder_and_bn(backbone_name=backbone_name, pretrained=pretrained_encoder)
        self.decoder = build_decoder(backbone_name=backbone_name)

        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        # super().train(mode) already recurses into encoder/bn/decoder; only the frozen teacher
        # needs re-pinning to eval() so its BatchNorm statistics never drift.
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, batch: torch.Tensor) -> list[torch.Tensor]:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(batch)
        return self.encoder(batch)

    def _forward_features(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        teacher_features = self._encode(batch)
        bottleneck_features = self.bn(teacher_features)
        student_features = self.decoder(bottleneck_features)
        return teacher_features, bottleneck_features, student_features

    def forward(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """Training forward pass: returns ``(teacher_features, bottleneck, student_features)``."""
        return self._forward_features(batch)

    def inference(self, batch: torch.Tensor) -> torch.Tensor:
        """Return the fused per-pixel anomaly map ``(B, H, W)`` at ``input_size``."""
        teacher_features, _, student_features = self._forward_features(batch)
        return self._anomaly_map(teacher_features, student_features)

    def _anomaly_map(
        self,
        teacher_features: list[torch.Tensor],
        student_features: list[torch.Tensor],
    ) -> torch.Tensor:
        anomaly_map: torch.Tensor | None = None

        for teacher_feature, student_feature in zip(teacher_features, student_features):
            component_map = 1.0 - F.cosine_similarity(student_feature, teacher_feature, dim=1)
            component_map = component_map.unsqueeze(1)
            component_map = F.interpolate(
                component_map,
                size=self.input_size,
                mode="bilinear",
                align_corners=True,
            )

            anomaly_map = component_map if anomaly_map is None else anomaly_map + component_map

        if anomaly_map is None:
            raise ValueError("RD4AD received empty feature lists during anomaly map computation")

        anomaly_map = self._smooth(anomaly_map)
        return anomaly_map[:, 0]

    def _smooth(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Gaussian-blur the anomaly map, deriving the kernel size from sigma like scipy."""
        if self.score_smoothing_sigma <= 0.0:
            return anomaly_map

        radius = int(self._SMOOTHING_TRUNCATE * self.score_smoothing_sigma + 0.5)
        kernel = 2 * radius + 1

        height, width = anomaly_map.shape[-2:]
        max_odd = min(height, width)
        max_odd -= 1 - (max_odd % 2)
        kernel = min(kernel, max_odd)
        if kernel <= 1:
            return anomaly_map

        sigma = [self.score_smoothing_sigma, self.score_smoothing_sigma]
        return gaussian_blur(anomaly_map, kernel_size=[kernel, kernel], sigma=sigma)

    @staticmethod
    def cosine_loss(teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]) -> torch.Tensor:
        loss = teacher_features[0].new_tensor(0.0)
        cosine_similarity = nn.CosineSimilarity(dim=1)

        for teacher_feature, student_feature in zip(teacher_features, student_features):
            loss = loss + torch.mean(
                1.0
                - cosine_similarity(
                    teacher_feature.view(teacher_feature.shape[0], -1),
                    student_feature.view(student_feature.shape[0], -1),
                )
            )

        return loss
