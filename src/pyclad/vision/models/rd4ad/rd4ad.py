from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from pyclad.vision.models.rd4ad.architecture import RD4ADArchitecture
from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.utilities.base_model import LightningVisionModel


class RD4AD(LightningVisionModel):
    """RD4AD (Reverse Distillation from One-Class Embedding) vision anomaly detector."""

    config: RD4ADConfig

    def __init__(self, config: Optional[RD4ADConfig] = None):
        super().__init__(config or RD4ADConfig())

    def _build_module(self) -> pl.LightningModule:
        network = RD4ADArchitecture(
            backbone_name=self.config.backbone_name,
            input_size=self.config.input_size,
            pretrained_encoder=self.config.pretrained_encoder,
            freeze_encoder=self.config.freeze_encoder,
            score_smoothing_sigma=self.config.score_smoothing_sigma,
        )
        return RD4ADModule(
            network=network,
            learning_rate=self.config.learning_rate,
            adam_betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        return self.module.network.inference(batch)

    def name(self) -> str:
        return "RD4AD"


class RD4ADModule(pl.LightningModule):
    def __init__(
        self,
        network: RD4ADArchitecture,
        learning_rate: float,
        adam_betas: tuple[float, float],
        weight_decay: float,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        teacher_features, _, student_features = self.network(x)
        loss = RD4ADArchitecture.cosine_loss(teacher_features, student_features)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = list(self.network.decoder.parameters()) + list(self.network.bn.parameters())
        if not self.network.freeze_encoder:
            parameters = list(self.network.encoder.parameters()) + parameters

        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
