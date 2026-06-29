from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.vision.models.stfpm.backbones import (
    TorchvisionFeatureExtractor,
    default_stfpm_return_nodes,
)


class STFPMArchitecture(nn.Module):

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone_name: str,
        backbone_return_nodes: Optional[tuple[str, ...]],
        pretrained_teacher: bool,
        pretrained_student: bool,
        freeze_teacher: bool,
    ):
        super().__init__()

        if backbone_return_nodes is None:
            backbone_return_nodes = tuple(default_stfpm_return_nodes(backbone_name))

        self.input_size = tuple(input_size)
        self.freeze_teacher = freeze_teacher
        self.return_nodes = tuple(backbone_return_nodes)

        self.teacher = TorchvisionFeatureExtractor(
            backbone_name=backbone_name,
            return_nodes=self.return_nodes,
            pretrained=pretrained_teacher,
            freeze=freeze_teacher,
        )
        self.student = TorchvisionFeatureExtractor(
            backbone_name=backbone_name,
            return_nodes=self.return_nodes,
            pretrained=pretrained_student,
            freeze=False,
        )

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
                teacher_features = self.teacher(x)
        else:
            teacher_features = self.teacher(x)
        student_features = self.student(x)
        return teacher_features, student_features

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns ``(teacher_features, student_features)``. Use :meth:`inference` for the map."""
        return self._feature_pairs(x)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        teacher_features, student_features = self._feature_pairs(x)
        return self.anomaly_map(teacher_features, student_features)

    def anomaly_map(
        self,
        teacher_features: list[torch.Tensor],
        student_features: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(teacher_features) == 0 or len(student_features) == 0:
            raise ValueError("STFPM anomaly map requires non-empty teacher and student feature lists")

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

        return aggregated_map[:, 0]

    @staticmethod
    def feature_loss(teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]) -> torch.Tensor:
        if len(teacher_features) == 0 or len(student_features) == 0:
            raise ValueError("STFPM feature loss requires non-empty teacher and student feature lists")

        loss: Optional[torch.Tensor] = None
        for teacher_feature, student_feature in zip(teacher_features, student_features):
            teacher_feature = F.normalize(teacher_feature, dim=1)
            student_feature = F.normalize(student_feature, dim=1)
            layer_loss = torch.sum((teacher_feature - student_feature) ** 2, dim=1).mean()
            loss = layer_loss if loss is None else loss + layer_loss
        return loss
