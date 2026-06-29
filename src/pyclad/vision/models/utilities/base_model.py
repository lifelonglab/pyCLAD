from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

from pyclad.vision.models.utilities.config import LightningVisionConfig, VisionConfig
from pyclad.vision.models.utilities.preprocessing import ImagePreprocessor
from pyclad.vision.models.utilities.utils import (
    BestWeightsCallback,
    resolve_device,
    to_float,
    trainer_device_config,
)
from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.prediction_results import VisionPredictionResults


class VisionScoringBase(VisionModel):
    """Shared preprocessing, scoring, prediction and threshold logic for vision AD models.

    Subclasses must implement :meth:`fit`, :meth:`name` and :meth:`_inference_maps`. The
    inference hook is the single seam every model plugs into: it must return per-pixel
    anomaly maps of shape ``(B, H, W)`` following the convention *higher = more anomalous*.
    Everything else (batching, image-level aggregation, map resizing, threshold calibration
    and resolution) is provided here and is independent of the training strategy.
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self._device = resolve_device(config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=config.input_size,
            in_channels=3,
            input_range=config.input_range,
            input_layout=config.input_layout,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        )
        self._threshold: Optional[float] = config.threshold
        self._last_loss: Optional[float] = None

    # --- subclass contract ---------------------------------------------------
    @abstractmethod
    def _inference_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """Return per-pixel anomaly maps ``(B, H, W)``; higher values are more anomalous.

        ``batch`` is a preprocessed float tensor on this model's device.

        For :class:`LightningVisionModel` subclasses this is invoked inside an eval-mode,
        ``no_grad`` context managed by :meth:`LightningVisionModel._run_inference` (which also
        restores the prior training mode), so the implementation should be a plain forward
        pass. Single-pass (non-Lightning) models manage any grad/eval context themselves.
        """

    def _extra_info(self) -> dict:
        """Model-specific entries merged into :meth:`additional_info`."""
        return {}

    def _apply_seed(self) -> None:
        """Seed the torch global RNG so weight init and channel permutations are reproducible.

        A no-op when ``config.seed is None``. Concrete models must call this immediately
        before their stochastic initialization: :class:`LightningVisionModel` does so before
        building its module; single-pass models (e.g. memory-bank) should call it before
        their random setup in ``fit``. Data-shuffling reproducibility is handled separately
        (and universally) by the seeded generator in :meth:`_prepare_batches`.

        Only the torch RNG is touched — intentionally not ``random``/``numpy``/env vars
        (as ``pl.seed_everything`` would), since torch covers init and permutations here.
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

    def _prepare_batches(self, data: np.ndarray, shuffle: bool) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        generator = torch.Generator().manual_seed(self.config.seed) if self.config.seed is not None else None
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=0, generator=generator
        )

    def _aggregate_scores(self, score_maps: torch.Tensor) -> torch.Tensor:
        if self.config.score_mode == "mean":
            return score_maps.mean(dim=(1, 2))
        return score_maps.reshape(score_maps.size(0), -1).max(dim=1).values

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Resize continuous per-pixel anomaly scores to ``output_size`` via bilinear."""
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps[:, None, :, :], size=output_size, mode="bilinear", align_corners=False)
        return resized[:, 0]

    def _run_inference(self, batch_x: torch.Tensor) -> torch.Tensor:
        return self._inference_maps(batch_x.to(self._device, dtype=torch.float32))

    def _score_data(self, data: np.ndarray) -> np.ndarray:
        """Compute image-level anomaly scores. Used internally for threshold calibration."""
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        all_scores: list[np.ndarray] = []
        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            batch_scores = self._aggregate_scores(self._run_inference(batch_x))
            all_scores.append(batch_scores.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(all_scores, axis=0) if all_scores else np.asarray([], dtype=np.float32)

    def _calibrate_threshold(self, data: np.ndarray) -> None:
        """Set the decision threshold from a fixed config value or a quantile of training scores."""
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
            return
        scores = self._score_data(data)
        self._threshold = float(np.quantile(scores, self.config.threshold_quantile)) if len(scores) > 0 else None

    def _resolve_threshold(self) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        if self._threshold is not None:
            return float(self._threshold)
        raise RuntimeError(
            f"{self.name()}: anomaly threshold is not calibrated. Call fit() on nominal data "
            "or set config.threshold explicitly before predict()."
        )

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        if len(data) == 0:
            empty = np.asarray([], dtype=np.float32)
            return VisionPredictionResults(
                y_pred=np.asarray([], dtype=np.int64), anomaly_scores=empty, score_maps=empty
            )

        target_size = self._preprocessor.spatial_size(data)
        all_maps: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            raw_maps = self._run_inference(batch_x)

            batch_scores = self._aggregate_scores(raw_maps)
            all_scores.append(batch_scores.detach().cpu().numpy().astype(np.float32, copy=False))

            resized = self._resize_maps(raw_maps, output_size=target_size)
            all_maps.append(resized.detach().cpu().numpy().astype(np.float32, copy=False))

        score_maps = np.concatenate(all_maps, axis=0)
        anomaly_scores = np.concatenate(all_scores, axis=0)
        threshold = self._resolve_threshold()

        return VisionPredictionResults(
            y_pred=(anomaly_scores > threshold).astype(int),
            anomaly_scores=anomaly_scores,
            score_maps=score_maps,
        )

    def additional_info(self) -> dict:
        return {
            **self.config.model_dump(),
            **self._extra_info(),
            "threshold": self._threshold,
            "last_loss": self._last_loss,
        }


class LightningVisionModel(VisionScoringBase):
    """Base for vision AD models trained end-to-end with a PyTorch Lightning trainer.

    Adds the gradient-training :meth:`fit` (optimizer loop, optional early stopping with
    best-weight restoration, threshold calibration). Subclasses provide the network via
    :meth:`_build_module` and the inference hook via :meth:`_inference_maps`.
    """

    config: LightningVisionConfig

    def __init__(self, config: LightningVisionConfig):
        super().__init__(config)
        self._apply_seed()  # before _build_module(): seeds weight init and channel permutations
        self.module = self._build_module().to(self._device)

    @abstractmethod
    def _build_module(self) -> pl.LightningModule:
        """Build the LightningModule. It must expose the trainable network as ``.network``."""

    def _run_inference(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Run inference in eval mode under ``no_grad`` and restore the prior training mode.

        Switching to ``eval()`` is required for correct inference (frozen BatchNorm stats,
        disabled Dropout), but the previous mode must be restored afterwards: pyCLAD reuses
        the same model instance across the concept stream, and PyTorch Lightning (>=2.2) no
        longer resets train mode at the start of ``fit()``. A leaked ``eval()`` would silently
        train BatchNorm/Dropout submodules in eval mode on every fit after the first.
        ``train(was_training)`` re-dispatches into the architecture's overridden ``train()``,
        so frozen backbones remain pinned to eval.
        """
        was_training = self.module.training
        self.module.eval()
        try:
            with torch.no_grad():
                return super()._run_inference(batch_x)
        finally:
            self.module.train(was_training)

    def fit(self, data: np.ndarray):
        if len(data) == 0 or self.config.epochs == 0:
            return

        callbacks: list[pl.Callback] = []
        best_weights_callback: Optional[BestWeightsCallback] = None

        if self.config.early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="train_loss",
                    mode="min",
                    patience=self.config.early_stopping_patience,
                    min_delta=float(self.config.early_stopping_min_delta),
                    check_on_train_epoch_end=True,
                )
            )

            if self.config.early_stopping_restore_best:
                best_weights_callback = BestWeightsCallback(
                    monitor="train_loss",
                    min_delta=float(self.config.early_stopping_min_delta),
                )
                callbacks.append(best_weights_callback)

        accelerator, devices = trainer_device_config(self._device)
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=self.config.show_training_progress,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

        trainer.fit(self.module, train_dataloaders=self._prepare_batches(data, shuffle=True))
        self.module = self.module.to(self._device)
        self._last_loss = to_float(trainer.callback_metrics.get("train_loss"))

        if (
            self.config.early_stopping_patience is not None
            and self.config.early_stopping_restore_best
            and best_weights_callback is not None
            and best_weights_callback.best_state_dict is not None
        ):
            self.module.network.load_state_dict(best_weights_callback.best_state_dict)

        self._calibrate_threshold(data)
