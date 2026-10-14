"""COMMAND's supervised complete-video loss."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from pyclad.video.models.command.architecture import CommandOutput
from pyclad.video.models.command.config import CommandLossConfig


@dataclass(frozen=True)
class CommandLossBreakdown:
    total: Tensor
    mil: Tensor
    contrastive: Tensor
    focal: Tensor
    anomaly_separation: Tensor
    sparsity: Tensor
    smoothness: Tensor
    score_l2: Tensor

    def detached(self) -> Dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.__dict__.items()}


def command_composite_loss(
    output: CommandOutput,
    weak_labels: Tensor,
    config: CommandLossConfig | None = None,
) -> CommandLossBreakdown:
    """Compute COMMAND's MIL, contrastive, focal, and separation objective."""

    config = config or CommandLossConfig()
    labels = weak_labels.to(device=output.logits.device, dtype=output.logits.dtype).reshape(-1)
    if output.logits.ndim != 2 or len(labels) != output.logits.shape[0]:
        raise ValueError("weak_labels must provide one value per video bag")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("weak_labels must contain only zero and one")

    scores = output.memory.dual_memory_deviation
    bag_scores = scores.amax(dim=1)
    positive = labels > 0.5
    normal = ~positive
    zero = scores.sum() * 0.0

    if positive.any() and normal.any():
        ranking = F.relu(config.mil_margin - bag_scores[positive, None] + bag_scores[None, normal]).mean()
        separation = F.relu(
            config.anomaly_margin - scores[positive].mean(dim=1, keepdim=True) + scores[normal].mean(dim=1).unsqueeze(0)
        ).mean()
    else:
        ranking = zero
        separation = zero

    sparsity = scores[positive].mean() if positive.any() else zero
    smoothness = (scores[:, 1:] - scores[:, :-1]).square().mean() if scores.shape[1] > 1 else zero
    mil = ranking + config.sparsity_weight * sparsity + config.smoothness_weight * smoothness

    projections = F.normalize(output.projections.mean(dim=1), dim=-1)
    contrastive = (
        _supervised_info_nce(projections, labels, config.contrastive_temperature)
        if positive.any() and normal.any()
        else zero
    )

    cross_entropy = F.binary_cross_entropy_with_logits(output.logits.amax(dim=1), labels, reduction="none")
    positive_probability = torch.exp(-cross_entropy)
    alpha = torch.where(labels > 0.5, config.focal_alpha, 1.0 - config.focal_alpha)
    focal = (alpha * (1.0 - positive_probability).pow(config.focal_gamma) * cross_entropy).mean()
    score_l2 = scores.square().mean()

    total = (
        config.mil_weight * mil
        + config.contrastive_weight * contrastive
        + config.focal_weight * focal
        + config.anomaly_separation_weight * separation
        + config.score_l2_weight * score_l2
    )
    return CommandLossBreakdown(
        total=total,
        mil=mil,
        contrastive=contrastive,
        focal=focal,
        anomaly_separation=separation,
        sparsity=sparsity,
        smoothness=smoothness,
        score_l2=score_l2,
    )


def _supervised_info_nce(projections: Tensor, labels: Tensor, temperature: float) -> Tensor:
    if len(projections) < 2:
        return projections.sum() * 0.0
    logits = projections @ projections.transpose(0, 1) / temperature
    identity = torch.eye(len(projections), dtype=torch.bool, device=projections.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return projections.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits.masked_fill(identity, -torch.inf), dim=1, keepdim=True)
    mean_positive_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_positive_log_prob[valid].mean()
