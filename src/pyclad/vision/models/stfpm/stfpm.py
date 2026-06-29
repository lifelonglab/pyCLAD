from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from pyclad.vision.models.stfpm.architecture import STFPMArchitecture
from pyclad.vision.models.stfpm.config import STFPMConfig
from pyclad.vision.models.utilities.base_model import LightningVisionModel


class STFPM(LightningVisionModel):
    def __init__(self, config: Optional[STFPMConfig] = None):
        super().__init__(config or STFPMConfig())

    def _build_module(self) -> pl.LightningModule:
        network = STFPMArchitecture(
            input_size=self.config.input_size,
            backbone_name=self.config.backbone_name,
            backbone_return_nodes=self.config.backbone_return_nodes,
            pretrained_teacher=self.config.pretrained_teacher,
            pretrained_student=self.config.pretrained_student,
            freeze_teacher=self.config.freeze_teacher,
        )
        return STFPMModule(
            network=network,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        return self.module.network.inference(batch)

    def _extra_info(self) -> dict:
        return {"backbone_return_nodes": self.module.network.return_nodes}

    def name(self) -> str:
        return "STFPM"


class STFPMModule(pl.LightningModule):
    def __init__(
        self,
        network: STFPMArchitecture,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        teacher_features, student_features = self.network(x)
        loss = STFPMArchitecture.feature_loss(teacher_features, student_features)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Canonical STFPM optimizes only the student; the teacher stays frozen.
        if self.network.freeze_teacher:
            params = self.network.student.parameters()
        else:
            params = self.network.parameters()
        return torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
