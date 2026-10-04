from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn import functional

from pyclad.vision.models.continual_mega_baseline.config import (
    ContinualMegaBaselineConfig,
)
from pyclad.vision.models.continual_mega_baseline.losses import (
    binary_dice_loss,
    focal_loss,
)
from pyclad.vision.models.continual_mega_baseline.prompt_learner import PromptLearner
from pyclad.vision.models.continual_mega_baseline.visual_encoder import (
    AdaptedVisualEncoder,
)

NORMAL_CHANNEL = 0
ANOMALY_CHANNEL = 1


class ContinualMegaBaselineTrainer:
    def __init__(
        self,
        visual_encoder: AdaptedVisualEncoder,
        prompt_learner: PromptLearner,
        text_features: Callable[[], torch.Tensor],
        config: ContinualMegaBaselineConfig,
        train_prompts: bool,
    ):
        self.visual_encoder = visual_encoder
        self.prompt_learner = prompt_learner
        self.text_features = text_features
        self.config = config
        self.epochs_completed = 0

        self.adapter_optimizer = torch.optim.AdamW(visual_encoder.adapters.parameters(), lr=config.learning_rate)
        self.prompt_optimizer: Optional[torch.optim.Optimizer] = None
        if train_prompts:
            self.prompt_optimizer = torch.optim.Adam(
                prompt_learner.parameters(), lr=config.prompt_learning_rate, betas=(0.5, 0.999)
            )

    def train_epoch(self, images: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> float:
        generator = torch.Generator().manual_seed(self.config.seed + self.epochs_completed)
        order = torch.randperm(len(images), generator=generator)

        losses = []
        for start in range(0, len(order), self.config.train_batch_size):
            batch = order[start : start + self.config.train_batch_size]
            losses.append(self._train_step(images[batch], labels[batch], masks[batch]))

        self.epochs_completed += 1
        return sum(losses) / len(losses)

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "epochs_completed": self.epochs_completed,
            "adapters": self.visual_encoder.adapters.state_dict(),
            "prompt_state_dict": self.prompt_learner.state_dict(),
            "adapter_optimizer": self.adapter_optimizer.state_dict(),
        }
        if self.prompt_optimizer is not None:
            state["prompt_optimizer"] = self.prompt_optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.epochs_completed = state["epochs_completed"]
        self.visual_encoder.adapters.load_state_dict(state["adapters"])
        self.prompt_learner.load_state_dict(state["prompt_state_dict"])
        self.adapter_optimizer.load_state_dict(state["adapter_optimizer"])
        if self.prompt_optimizer is not None:
            self.prompt_optimizer.load_state_dict(state["prompt_optimizer"])

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> float:
        device = next(self.visual_encoder.parameters()).device
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)

        with torch.autocast("cuda"):
            tokens, noisy_tokens = self.visual_encoder(images, with_noise=self.config.use_synthetic_anomalies)
            text_features = self.text_features()
            loss = self._segmentation_loss(tokens, text_features, labels, masks)
            loss = loss + self._synthetic_anomaly_loss(noisy_tokens, text_features, labels)

        self.adapter_optimizer.zero_grad()
        if self.prompt_optimizer is not None:
            self.prompt_optimizer.zero_grad()
        loss.backward()
        self.adapter_optimizer.step()
        if self.prompt_optimizer is not None:
            self.prompt_optimizer.step()
        return loss.item()

    def _segmentation_loss(
        self,
        tokens: List[torch.Tensor],
        text_features: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        normal, anomalous = labels == 0, labels == 1
        loss = torch.zeros((), device=text_features.device)

        for layer_tokens in tokens:
            logits = _patch_logits(layer_tokens, text_features, self.config.logit_scale)
            if normal.any():
                loss = loss + functional.cross_entropy(
                    logits[normal].reshape(-1, 2),
                    torch.full((int(normal.sum()) * logits.shape[1],), NORMAL_CHANNEL, device=logits.device),
                )
            if anomalous.any():
                probabilities = _spatial_probabilities(logits, self.config.image_size)
                loss = loss + focal_loss(probabilities[anomalous], masks[anomalous])
                loss = loss + binary_dice_loss(probabilities[anomalous, ANOMALY_CHANNEL], masks[anomalous])
        return loss

    def _synthetic_anomaly_loss(
        self,
        noisy_tokens: List[torch.Tensor],
        text_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        normal = labels == 0
        loss = torch.zeros((), device=text_features.device)
        if not normal.any():
            return loss

        for layer_tokens in noisy_tokens:
            logits = _patch_logits(layer_tokens, text_features, self.config.logit_scale)
            loss = loss + functional.cross_entropy(
                logits[normal].reshape(-1, 2),
                torch.full((int(normal.sum()) * logits.shape[1],), ANOMALY_CHANNEL, device=logits.device),
            )
        return loss


def _patch_logits(tokens: torch.Tensor, text_features: torch.Tensor, logit_scale: float) -> torch.Tensor:
    return logit_scale * functional.normalize(tokens, dim=-1) @ text_features


def _spatial_probabilities(logits: torch.Tensor, image_size: int) -> torch.Tensor:
    batch, patches, channels = logits.shape
    side = int(patches**0.5)
    maps = functional.interpolate(
        logits.permute(0, 2, 1).view(batch, channels, side, side),
        size=image_size,
        mode="bilinear",
        align_corners=True,
    )
    return maps.softmax(dim=1)
