from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from pyclad.vision.models.fastflow.architecture import FastFlowArchitecture
from pyclad.vision.models.fastflow.config import FastFlowConfig
from pyclad.vision.models.utilities.base_model import LightningVisionModel


class FastFlow(LightningVisionModel):
    def __init__(self, config: Optional[FastFlowConfig] = None):
        super().__init__(config or FastFlowConfig())

    def _build_module(self) -> pl.LightningModule:
        network = FastFlowArchitecture(
            input_size=self.config.input_size,
            backbone_name=self.config.backbone_name,
            backbone_return_nodes=self.config.backbone_return_nodes,
            pretrained_backbone=self.config.pretrained_backbone,
            freeze_backbone=self.config.freeze_backbone,
            normalize_features=self.config.normalize_features,
            flow_steps=self.config.flow_steps,
            conv3x3_only=self.config.conv3x3_only,
            hidden_ratio=self.config.hidden_ratio,
            affine_clamping=self.config.affine_clamping,
        )
        return FastFlowModule(
            network=network,
            learning_rate=self.config.learning_rate,
            adam_betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        return self.module.network.inference(batch)

    def _extra_info(self) -> dict:
        return {"backbone_return_nodes": self.module.network.return_nodes}

    def name(self) -> str:
        return "FastFlow"


class FastFlowModule(pl.LightningModule):
    def __init__(
        self,
        network: FastFlowArchitecture,
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
        hidden_variables, log_jacobians = self.network(x)
        loss = FastFlowArchitecture.fastflow_loss(hidden_variables, log_jacobians)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
