from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.models.vision.paste.backbones import (
    PaSTeBackbone,
    default_ad_layers,
    supported_backbone_names,
)


class PaSTeArchitecture(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        ad_layers: tuple[int, ...] | None,
        student_bootstrap_layer: int | None,
        pretrained_teacher: bool,
        pretrained_student: bool,
        freeze_teacher: bool,
        input_size: tuple[int, int],
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.ad_layers = tuple(sorted(ad_layers if ad_layers is not None else default_ad_layers(backbone_name)))
        self.student_bootstrap_layer = student_bootstrap_layer
        self.input_size = tuple(input_size)
        self.freeze_teacher = freeze_teacher

        self.teacher = PaSTeBackbone(
            backbone_name=backbone_name,
            ad_layers=self.ad_layers,
            pretrained=pretrained_teacher,
            freeze=freeze_teacher,
            bootstrap_layer=self.student_bootstrap_layer,
            is_teacher=True,
        )
        self.student = PaSTeBackbone(
            backbone_name=backbone_name,
            ad_layers=self.ad_layers,
            pretrained=pretrained_student,
            freeze=False,
            bootstrap_layer=self.student_bootstrap_layer,
            is_teacher=False,
        )

    @property
    def return_nodes(self) -> tuple[int, ...]:
        return self.ad_layers

    @staticmethod
    def supported_backbone_names() -> tuple[str, ...]:
        return supported_backbone_names()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_teacher:
            self.teacher.eval()
        else:
            self.teacher.train(mode)
        self.student.train(mode)
        return self

    def _feature_pairs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.freeze_teacher:
            self.teacher.eval()
            with torch.no_grad():
                teacher_features, bootstrap_feature = self.teacher(x)
        else:
            teacher_features, bootstrap_feature = self.teacher(x)

        student_input = x if self.student_bootstrap_layer is None else bootstrap_feature
        if student_input is None:
            raise RuntimeError("PaSTe failed to produce the student bootstrap feature")
        student_features, _ = self.student(student_input)
        return teacher_features, student_features

    def forward_train(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not self.training:
            self.train()
        return self._feature_pairs(x)

    def forward(self, x: torch.Tensor):
        teacher_features, student_features = self._feature_pairs(x)
        if self.training:
            return teacher_features, student_features
        return self.post_process(teacher_features, student_features)

    def post_process(
        self,
        teacher_features: list[torch.Tensor],
        student_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aggregated_map: Optional[torch.Tensor] = None

        for teacher_feature, student_feature in zip(teacher_features, student_features):
            teacher_feature = F.normalize(teacher_feature, dim=1)
            student_feature = F.normalize(student_feature, dim=1)
            distance_map = torch.sum((teacher_feature - student_feature) ** 2, dim=1, keepdim=True)
            distance_map = F.interpolate(
                distance_map,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            aggregated_map = distance_map if aggregated_map is None else aggregated_map * distance_map

        if aggregated_map is None:
            raise ValueError("PaSTe received empty feature lists during post-processing")

        anomaly_map = aggregated_map[:, 0]
        anomaly_scores = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1).values
        return anomaly_map, anomaly_scores

    @staticmethod
    def feature_loss(teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]) -> torch.Tensor:
        if len(teacher_features) == 0 or len(student_features) == 0:
            raise ValueError("PaSTe feature loss requires non-empty teacher and student feature lists")

        loss = torch.sum((teacher_features[0] - student_features[0]) ** 2, dim=1).mean()
        for teacher_feature, student_feature in zip(teacher_features[1:], student_features[1:]):
            teacher_feature = F.normalize(teacher_feature, dim=1)
            student_feature = F.normalize(student_feature, dim=1)
            loss = loss + torch.sum((teacher_feature - student_feature) ** 2, dim=1).mean()
        return loss
