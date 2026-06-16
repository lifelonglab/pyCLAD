from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from pyclad.vision.models.paste.architecture import PaSTeArchitecture
from pyclad.vision.models.paste.config import PaSTeConfig
from pyclad.vision.models.utilities.base_model import LightningVisionModel


class PaSTe(LightningVisionModel):
    def __init__(self, config: Optional[PaSTeConfig] = None):
        super().__init__(config or PaSTeConfig())

    def _build_module(self) -> pl.LightningModule:
        network = PaSTeArchitecture(
            backbone_name=self.config.backbone_name,
            ad_layers=self.config.ad_layers,
            student_bootstrap_layer=self.config.student_bootstrap_layer,
            pretrained_teacher=self.config.pretrained_teacher,
            pretrained_student=self.config.pretrained_student,
            freeze_teacher=self.config.freeze_teacher,
            input_size=self.config.input_size,
        )
        return PaSTeModule(
            network=network,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        self.module.eval()
        with torch.no_grad():
            score_maps, _ = self.module.network.inference(batch)
        return score_maps

    def _extra_info(self) -> dict:
        return {"ad_layers": self.module.network.ad_layers}

    def name(self) -> str:
        return "PaSTe"


class PaSTeModule(pl.LightningModule):
    def __init__(
        self,
        network: PaSTeArchitecture,
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

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        teacher_features, student_features = self.network(x)
        loss = PaSTeArchitecture.feature_loss(teacher_features, student_features)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
