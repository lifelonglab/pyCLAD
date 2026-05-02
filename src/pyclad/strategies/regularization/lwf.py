"""Learning without Forgetting (LwF) strategy for continual learning."""

import copy
import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.neural_model import NeuralTrainableModel
from pyclad.strategies.neural_hooks import NeuralStrategyHooks
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptIncrementalStrategy,
)

logger = logging.getLogger(__name__)


class LwFStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """Learning without Forgetting with latent or reconstruction distillation."""

    def __init__(
        self,
        model: NeuralTrainableModel,
        alpha: float = 0.5,
        distill_mode: str = "latent",
        epochs: Optional[int] = None,
    ):
        self._model = model
        self._old_model: Optional[NeuralTrainableModel] = None
        self._alpha = alpha
        self._distill_mode = distill_mode
        self._task_count = 0
        self._hooks = NeuralStrategyHooks(model)
        self._hooks.validate_trainable_model("LwFStrategy")
        self._distill_fallback_warnings = set()
        self._epochs = self._resolve_epochs(epochs)
        self._old_batch_for_distillation: Optional[torch.Tensor] = None

        if distill_mode not in ["latent", "reconstruction", "hybrid"]:
            raise ValueError(
                f"Invalid distill_mode: {distill_mode}. " f"Must be 'latent', 'reconstruction', or 'hybrid'"
            )

    def _resolve_epochs(self, epochs: Optional[int]) -> int:
        resolved_epochs = self._hooks.epochs() if epochs is None else int(epochs)
        if resolved_epochs <= 0:
            raise ValueError(f"epochs must be positive, got {resolved_epochs}")
        return resolved_epochs

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        if self._old_model is None:
            logger.info(f"Task {self._task_count + 1}: Training without distillation (first task)")
            self._model.fit(data)
        else:
            logger.info(
                f"Task {self._task_count + 1}: Training with LwF distillation "
                f"(alpha={self._alpha}, mode={self._distill_mode})"
            )
            self._fit_with_distillation(data)

        self._task_count += 1
        self._old_model = self._clone_model()

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple:
        return self._hooks.predict(data)

    def name(self) -> str:
        return "LwF"

    def additional_info(self) -> Dict:
        return {
            "model": self._model.name(),
            "alpha": self._alpha,
            "distill_mode": self._distill_mode,
            "task_count": self._task_count,
            "has_old_model": self._old_model is not None,
            "epochs": self._epochs,
        }

    def _resolve_shuffle(self) -> bool:
        return self._hooks.shuffle()

    def _supports_latent_distillation(self, model: Optional[NeuralTrainableModel] = None) -> bool:
        target = self._model if model is None else model
        if callable(getattr(target, "encode_batch", None)):
            return True

        module = self._hooks.resolve_module(model)
        if module is None:
            return False

        return hasattr(module, "encode") or hasattr(module, "encoder")

    @staticmethod
    def _extract_latent_representation(encoded_output):
        if isinstance(encoded_output, (tuple, list)):
            for item in encoded_output:
                if torch.is_tensor(item):
                    return item
        return encoded_output

    def _encode_with_model(self, model: NeuralTrainableModel, batch: torch.Tensor) -> torch.Tensor:
        return self._extract_latent_representation(self._hooks.encode_batch(batch, model))

    def _forward_with_model(
        self, model: NeuralTrainableModel, batch: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], None]:
        return self._hooks.forward_batch(batch, model)

    def _training_batch_step(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None]:
        if hasattr(self._model, "training_batch_step"):
            return self._model.training_batch_step(batch)

        x_hat, z, _ = self._forward_with_model(self._model, batch)
        module = self._hooks.resolve_module(self._model)
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for distillation.")
        task_loss = self._hooks.batch_loss(module, batch)
        return task_loss, x_hat, z, None

    def _resolve_distill_mode(self) -> str:
        if self._distill_mode in ["latent", "hybrid"] and not (
            self._supports_latent_distillation(self._model) and self._supports_latent_distillation(self._old_model)
        ):
            logger.warning(
                "Latent distillation requested, but the model does not expose encoder hooks. "
                "Falling back to reconstruction distillation."
            )
            return "reconstruction"

        return self._distill_mode

    def _clone_model(self) -> NeuralTrainableModel:
        cloned = copy.deepcopy(self._model)

        module = self._hooks.resolve_module(cloned)
        if module is not None:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        return cloned

    def _compute_distill_loss(
        self,
        batch: torch.Tensor,
        distill_mode: str,
        x_hat_new: torch.Tensor,
        z_new: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        losses = []
        old_batch = self._old_batch_for_distillation
        if old_batch is None:
            old_batch = batch

        if distill_mode in {"latent", "hybrid"} and z_new is not None:
            try:
                with torch.no_grad():
                    z_old = self._encode_with_model(self._old_model, old_batch)
                if z_old.shape == z_new.shape:
                    losses.append(torch.nn.functional.mse_loss(z_new, z_old))
            except TypeError:
                pass

        if distill_mode in {"reconstruction", "hybrid"} or not losses:
            try:
                x_hat_current = self._current_reconstruction_for_distillation(batch, x_hat_new)
                with torch.no_grad():
                    x_hat_old = self._deterministic_reconstruction(self._old_model, old_batch)
                if x_hat_old.shape == x_hat_current.shape:
                    losses.append(torch.nn.functional.mse_loss(x_hat_current, x_hat_old))
            except (TypeError, ValueError):
                pass

        if not losses and z_new is not None:
            try:
                with torch.no_grad():
                    z_old = self._encode_with_model(self._old_model, old_batch)
                if z_old.shape == z_new.shape:
                    losses.append(torch.nn.functional.mse_loss(z_new, z_old))
            except TypeError:
                pass

        if losses:
            return sum(losses) / len(losses)

        warning_key = (self._model.__class__.__name__, distill_mode)
        if warning_key not in self._distill_fallback_warnings:
            logger.warning(
                "No compatible LwF distillation channel is available for %s in mode '%s'. "
                "Training this task with the model loss only.",
                self._model.__class__.__name__,
                distill_mode,
            )
            self._distill_fallback_warnings.add(warning_key)
        return None

    def _current_reconstruction_for_distillation(self, batch: torch.Tensor, x_hat_new: torch.Tensor) -> torch.Tensor:
        module = self._hooks.resolve_module(self._model)
        if self._uses_stochastic_variational_path(module):
            return self._deterministic_reconstruction(self._model, batch)
        return x_hat_new

    def _deterministic_reconstruction(self, model: NeuralTrainableModel, batch: torch.Tensor) -> torch.Tensor:
        module = self._hooks.resolve_module(model)
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for reconstruction distillation.")

        if self._uses_stochastic_variational_path(module):
            encoded = module.encoder(batch)
            latent = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
            return module.decoder(latent)

        x_hat, _, _ = self._forward_with_model(model, batch)
        return x_hat

    @staticmethod
    def _uses_stochastic_variational_path(module: Optional[torch.nn.Module]) -> bool:
        return module is not None and hasattr(module, "encoder") and hasattr(module, "decoder")

    def _weight_distill_loss(self, rec_loss: torch.Tensor, distill_loss: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if distill_loss is None:
            return None
        if self._alpha == 0:
            return distill_loss * 0.0

        scale = rec_loss.detach().abs() / distill_loss.detach().abs().clamp_min(1e-8)
        return self._alpha * scale * distill_loss

    def _fit_with_distillation(self, data: np.ndarray) -> None:
        self._hooks.prepare_fit(data)
        module = self._hooks.resolve_module(self._model)
        old_module = self._hooks.resolve_module(self._old_model)
        if module is None or old_module is None:
            logger.warning(
                "Model does not expose a trainable torch module required for distillation. "
                "Falling back to standard fit()."
            )
            self._model.fit(data)
            return

        tensor_data = self._hooks.prepare_data(data, self._model)
        old_tensor_data = self._hooks.prepare_data(data, self._old_model)
        dataset = TensorDataset(tensor_data, old_tensor_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self._hooks.batch_size(),
            shuffle=self._resolve_shuffle(),
            drop_last=self._hooks.drop_last(),
            num_workers=0,
        )
        if len(dataloader) == 0:
            logger.warning("Skipping LwF fit because no batches are available after batching.")
            return

        distill_mode = self._resolve_distill_mode()
        device = self._hooks.resolve_device(self._model)

        old_module.eval()
        for param in old_module.parameters():
            param.requires_grad = False

        old_module.to(device)
        module.to(device)
        module.train()

        optimizer = torch.optim.Adam(module.parameters(), lr=self._hooks.lr())

        epochs = self._epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_distill_loss = 0.0

            for batch_idx, (batch, old_batch) in enumerate(dataloader):
                batch = batch.to(device)
                old_batch = old_batch.to(device)

                rec_loss, x_hat_train, z_train, _ = self._training_batch_step(batch)
                x_hat_new, z_new = x_hat_train, z_train

                self._old_batch_for_distillation = old_batch
                try:
                    distill_loss = self._compute_distill_loss(batch, distill_mode, x_hat_new, z_new)
                finally:
                    self._old_batch_for_distillation = None

                weighted_distill_loss = self._weight_distill_loss(rec_loss, distill_loss)
                total_loss = rec_loss if weighted_distill_loss is None else rec_loss + weighted_distill_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_rec_loss += rec_loss.item()
                if distill_loss is not None:
                    epoch_distill_loss += distill_loss.item()

            n_batches = len(dataloader)
            avg_loss = epoch_loss / n_batches
            avg_rec = epoch_rec_loss / n_batches
            avg_distill = epoch_distill_loss / n_batches

            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: " f"Loss={avg_loss:.6f} (Rec={avg_rec:.6f}, Distill={avg_distill:.6f})"
                )

        self._hooks.after_fit(data)
