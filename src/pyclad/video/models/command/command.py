"""COMMAND backbone: network, supervised loss, calibration, and predictions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.video.data.sample import VideoBag, as_video_bags
from pyclad.video.models.command.architecture import CommandOutput, CommandVideoNetwork
from pyclad.video.models.command.config import CommandConfig
from pyclad.video.models.command.loss import (
    CommandLossBreakdown,
    command_composite_loss,
)
from pyclad.video.prediction_results import VideoPredictionResults


class CommandModel(TorchBackbone):
    """COMMAND model, independent of replay, task history, and training loops.

    Its base loss is supervised: callers supply one weak video label per bag
    in addition to the feature tensor. Unlabelled row loaders are not suitable
    for this model. ContTrain++ supplies labelled temporal batches.
    """

    def __init__(self, module: CommandVideoNetwork | None = None, config: CommandConfig | None = None):
        self.config = config or CommandConfig()
        torch.manual_seed(self.config.seed)
        self.module = module or CommandVideoNetwork(self.config.architecture)
        if self.module.architecture != self.config.architecture:
            raise ValueError("model architecture and configuration must match")
        self.to(torch.device(self.config.device))
        self._calibration_median = 0.0
        self._calibration_scale = 1.0
        self._threshold = self.config.threshold

    def get_module(self) -> nn.Module:
        """Return the underlying complete-video network."""
        return self.module

    def forward_output(self, x: Tensor, *, track_memory_usage: bool = False) -> CommandOutput:
        """Return network features and scores for a labelled-batch consumer."""
        return self.module(x, track_memory_usage=track_memory_usage)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw per-snippet dual-memory deviation scores."""
        return self.forward_output(x).memory.dual_memory_deviation

    def loss_breakdown(self, output: CommandOutput, weak_labels: Tensor) -> CommandLossBreakdown:
        """Evaluate the supervised base objective on an existing forward pass."""
        return command_composite_loss(output, weak_labels, self.config.loss)

    def compute_loss(self, x: Tensor, weak_labels: Tensor | None = None) -> Tensor:
        """Return a differentiable scalar loss for features and weak video labels."""
        if weak_labels is None:
            raise ValueError("COMMAND compute_loss requires one weak video label per bag")
        return self.loss_breakdown(self.forward_output(x), weak_labels).total

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return a fresh optimizer with the slower secondary-memory group."""
        secondary = []
        primary = []
        for name, parameter in self.module.named_parameters():
            if "secondary_memory" in name:
                secondary.append(parameter)
            else:
                primary.append(parameter)
        groups = [{"params": primary, "lr": self.config.learning_rate, "name": "primary"}]
        if secondary:
            groups.append(
                {
                    "params": secondary,
                    "lr": self.config.secondary_memory_learning_rate,
                    "name": "secondary_memory",
                }
            )
        return torch.optim.Adam(groups)

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        """Return bag maxima and aligned temporal scores for frame evaluation."""
        bags = as_video_bags(data)
        result = self.predict_bags(bags)
        scores = result["anomaly_scores"]
        maxima = scores.max(axis=1) if bags else np.empty(0)
        if bags and self._threshold is None:
            raise RuntimeError(
                "COMMAND: video threshold is not calibrated. Calibrate on normal bags or set config.threshold explicitly."
            )
        raw_maxima = result["raw_scores"].max(axis=1) if bags else np.empty(0)
        return VideoPredictionResults(
            y_pred=(raw_maxima > self._threshold).astype(np.int64) if bags else np.empty(0, dtype=np.int64),
            anomaly_scores=maxima,
            window_scores=scores.reshape(-1),
            classifier_scores=result["classifier_scores"].reshape(-1),
            windows=tuple(window for bag in bags for window in bag.windows),
        )

    def calibrate(self, normal_bags: Sequence[VideoBag], *, batch_size: int = 32) -> None:
        """Fit prediction normalization and the video threshold from normal bags.

        The caller chooses the calibration data; the model has no replay or
        historical-task knowledge. Empty data invalidates an inferred threshold.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        normal_bags = as_video_bags(normal_bags)
        if any(bag.weak_label != 0 for bag in normal_bags):
            raise ValueError("calibration requires normal video bags")
        self._threshold = self.config.threshold
        if not normal_bags:
            self._calibration_median = 0.0
            self._calibration_scale = 1.0
            return

        # Calibration videos may have different temporal lengths.
        by_length: Dict[int, list[VideoBag]] = {}
        for bag in normal_bags:
            by_length.setdefault(len(bag.features), []).append(bag)
        curves = []
        maxima = []
        for bags in by_length.values():
            raw = self.predict_bags(bags, batch_size=batch_size)["raw_scores"].astype(np.float64)
            curves.append(raw.reshape(-1))
            maxima.append(raw.max(axis=1))
        values = np.concatenate(curves)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        self._calibration_median = median
        self._calibration_scale = max(1e-8, 1.4826 * mad)
        if self._threshold is None:
            self._threshold = float(np.quantile(np.concatenate(maxima), self.config.threshold_quantile))

    def predict_bags(self, bags: Sequence[VideoBag], *, batch_size: int = 8) -> Dict[str, np.ndarray]:
        """Score complete videos in batches without modifying model state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        bags = as_video_bags(bags)
        if not bags:
            return {
                "anomaly_scores": np.empty((0, 0), dtype=np.float32),
                "raw_scores": np.empty((0, 0), dtype=np.float32),
                "classifier_scores": np.empty((0, 0), dtype=np.float32),
            }
        was_training = self.module.training
        self.module.eval()
        try:
            raw_blocks = []
            classifier_blocks = []
            with torch.no_grad():
                for offset in range(0, len(bags), batch_size):
                    batch = bags[offset : offset + batch_size]
                    features = torch.as_tensor(
                        np.stack([bag.features for bag in batch]),
                        dtype=torch.float32,
                        device=self.device(),
                    )
                    output = self.module(features)
                    raw_blocks.append(output.memory.dual_memory_deviation.cpu().numpy())
                    classifier_blocks.append(output.probabilities.cpu().numpy())
        finally:
            self.module.train(was_training)
        raw = np.concatenate(raw_blocks).astype(np.float32, copy=False)
        # A bounded monotonic map that preserves rank in extreme score tails.
        z = (raw.astype(np.float64) - self._calibration_median) / self._calibration_scale
        calibrated = 0.5 + np.arctan(z) / np.pi
        classifier = np.concatenate(classifier_blocks).astype(np.float32, copy=False)
        return {"anomaly_scores": calibrated, "raw_scores": raw, "classifier_scores": classifier}

    def state_dict(self) -> Mapping[str, object]:
        """Return model weights and prediction state, without training history."""
        return {
            "config": asdict(self.config),
            "module": self.module.state_dict(),
            "calibration_median": self._calibration_median,
            "calibration_scale": self._calibration_scale,
            "threshold": self._threshold,
            "distance_mode": "direct_euclidean",
        }

    def validate_state_dict(self, state: Mapping[str, object]) -> None:
        """Reject a checkpoint for a different model before loading its weights."""
        saved_config = {key: value for key, value in state["config"].items() if key != "device"}
        current_config = {key: value for key, value in asdict(self.config).items() if key != "device"}
        saved_config.setdefault("threshold", current_config["threshold"])
        saved_config.setdefault("threshold_quantile", current_config["threshold_quantile"])
        if saved_config != current_config:
            raise ValueError("checkpoint configuration does not match model configuration")

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore weights and prediction calibration on the current device."""
        self.validate_state_dict(state)
        self.module.load_state_dict(state["module"])
        self._calibration_median = float(state["calibration_median"])
        self._calibration_scale = float(state["calibration_scale"])
        self._threshold = state.get("threshold", self.config.threshold)

    def name(self) -> str:
        """Return the model's name."""
        return "COMMAND"

    def additional_info(self) -> Dict[str, object]:
        """Describe the model and scoring state separately from its strategy."""
        return {
            **asdict(self.config),
            "calibration": {
                "median": self._calibration_median,
                "mad_scale": self._calibration_scale,
                "threshold": self._threshold,
            },
        }
