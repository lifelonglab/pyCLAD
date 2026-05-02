"""Elastic Weight Consolidation (EWC) for torch-backed continual models."""

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


class EWCStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """Diagonal-Fisher EWC using the shared neural strategy hooks."""

    def __init__(
        self,
        model: NeuralTrainableModel,
        ewc_lambda: float = 1.0,
        epochs: Optional[int] = None,
        mode: str = "separate",
        decay_factor: float = 0.9,
        keep_importance_data: bool = True,
    ):
        self._model = model
        self._ewc_lambda = ewc_lambda
        self._mode = mode
        self._decay_factor = decay_factor
        self._keep_importance_data = keep_importance_data
        self._hooks = NeuralStrategyHooks(model)
        self._hooks.validate_trainable_model("EWCStrategy")
        self._loss_fn = None
        self._epochs = self._resolve_epochs(epochs)

        self._saved_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._importances: Dict[int, Dict[str, torch.Tensor]] = {}
        self._penalty_importance_sum: Dict[str, torch.Tensor] = {}
        self._penalty_reference_sum: Dict[str, torch.Tensor] = {}
        self._penalty_reference_square_sum: Dict[str, torch.Tensor] = {}
        self._task_count = 0

        if mode not in ["separate", "online"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'separate' or 'online'")

        if mode == "online":
            raise NotImplementedError("Online mode not yet implemented. Use 'separate' mode.")

    def _resolve_epochs(self, epochs: Optional[int]) -> int:
        resolved_epochs = self._hooks.epochs() if epochs is None else int(epochs)
        if resolved_epochs <= 0:
            raise ValueError(f"epochs must be positive, got {resolved_epochs}")
        return resolved_epochs

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        self._hooks.prepare_fit(data)
        if self._hooks.resolve_module() is None:
            raise TypeError(
                "EWCStrategy requires a PyTorch-backed model exposed via model.module_getter(), '.module', "
                "'.model', or '.model_'."
            )

        if self._task_count == 0:
            logger.info("Task %s: training without EWC penalty (first task)", self._task_count + 1)
        else:
            logger.info(
                "Task %s: training with EWC penalty (lambda=%s)",
                self._task_count + 1,
                self._ewc_lambda,
            )

        tensor_data = self._prepare_data(data)
        self._fit_with_ewc(tensor_data, data)
        self._hooks.after_fit(data)

        self._importances[self._task_count] = self._compute_fisher_information(tensor_data)
        self._saved_params[self._task_count] = self._get_current_params()
        self._rebuild_penalty_cache(device=self._hooks.resolve_device())

        if not self._keep_importance_data:
            self._keep_latest_task_only()

        self._task_count += 1

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple:
        return self._hooks.predict(data)

    def name(self) -> str:
        return "EWC"

    def additional_info(self) -> Dict:
        total_params = sum(sum(p.numel() for p in task_params.values()) for task_params in self._saved_params.values())
        total_importances = sum(
            sum(imp.numel() for imp in task_imps.values()) for task_imps in self._importances.values()
        )

        return {
            "model": self._model.name(),
            "ewc_lambda": self._ewc_lambda,
            "mode": self._mode,
            "task_count": self._task_count,
            "num_saved_tasks": len(self._saved_params),
            "total_stored_params": total_params,
            "total_stored_importances": total_importances,
            "memory_efficient": self._mode == "online",
            "epochs": self._epochs,
        }

    def _compute_ewc_penalty(self, module: torch.nn.Module) -> torch.Tensor:
        try:
            penalty = next(module.parameters()).new_tensor(0.0)
        except StopIteration:
            return torch.tensor(0.0)

        if not self._saved_params or not self._importances:
            return penalty

        if not self._penalty_importance_sum:
            self._rebuild_penalty_cache(device=penalty.device)

        for name, param in module.named_parameters():
            if not param.requires_grad or name not in self._penalty_importance_sum:
                continue

            importance_sum = self._penalty_importance_sum[name].to(device=param.device, dtype=param.dtype)
            reference_sum = self._penalty_reference_sum[name].to(device=param.device, dtype=param.dtype)
            reference_square_sum = self._penalty_reference_square_sum[name].to(device=param.device, dtype=param.dtype)
            penalty = penalty + (importance_sum * param.pow(2) - 2.0 * reference_sum * param + reference_square_sum).sum()

        return 0.5 * penalty

    def _fit_with_ewc(self, tensor_data: torch.Tensor, calibration_data: Optional[np.ndarray] = None) -> None:
        del calibration_data
        module = self._hooks.resolve_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for EWC.")

        if not torch.is_tensor(tensor_data):
            tensor_data = self._prepare_data(tensor_data)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self._hooks.batch_size(),
            shuffle=self._hooks.shuffle(),
            drop_last=self._hooks.drop_last(),
            num_workers=0,
        )
        if len(dataloader) == 0:
            logger.warning("Skipping EWC fit because no batches are available after batching.")
            return

        device = self._hooks.resolve_device()
        module.to(device)
        module.train()

        optimizer = torch.optim.Adam(module.parameters(), lr=self._hooks.lr())
        epochs = self._epochs

        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_task_loss = 0.0
            epoch_penalty = 0.0

            for (batch,) in dataloader:
                batch = batch.to(device)

                optimizer.zero_grad()
                task_loss = self._compute_batch_loss(module, batch)
                penalty = self._compute_ewc_penalty(module)
                total_loss = task_loss + self._ewc_lambda * penalty
                total_loss.backward()
                optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_penalty += penalty.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                n_batches = len(dataloader)
                logger.info(
                    "Epoch %s/%s: loss=%.6f (task=%.6f, ewc_penalty=%.6f)",
                    epoch + 1,
                    epochs,
                    epoch_total_loss / n_batches,
                    epoch_task_loss / n_batches,
                    epoch_penalty / n_batches,
                )

    def _compute_fisher_information(self, tensor_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        module = self._hooks.resolve_module()
        if module is None:
            raise TypeError("Model does not expose a trainable torch module required for Fisher estimation.")

        if not torch.is_tensor(tensor_data):
            tensor_data = self._prepare_data(tensor_data)
        dataset = TensorDataset(tensor_data)
        logger.info("Estimating EWC Fisher information with per-sample gradients.")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        device = self._hooks.resolve_device()
        module.to(device)

        was_training = module.training
        module.eval()

        fisher = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in module.named_parameters()
            if param.requires_grad
        }

        total_samples = 0

        for (batch,) in dataloader:
            batch = batch.to(device)
            batch_size = batch.shape[0]
            total_samples += batch_size

            module.zero_grad()
            loss = self._compute_batch_loss(module, batch)
            loss.backward()

            for name, param in module.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher[name] += param.grad.detach().cpu().pow(2) * batch_size

        if total_samples > 0:
            for name in fisher:
                fisher[name] /= total_samples

        module.zero_grad()
        if was_training:
            module.train()

        return {name: value.detach().clone() for name, value in fisher.items()}

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        return self._hooks.prepare_data(data)

    def _compute_batch_loss(self, module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        if self._loss_fn is not None:
            return NeuralStrategyHooks.ensure_scalar_loss(self._loss_fn(module, batch))
        return self._hooks.batch_loss(module, batch)

    @staticmethod
    def default_loss_fn(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return NeuralStrategyHooks.default_loss_fn(module, batch)

    def _rebuild_penalty_cache(self, device: Optional[torch.device] = None) -> None:
        self._penalty_importance_sum = {}
        self._penalty_reference_sum = {}
        self._penalty_reference_square_sum = {}
        target_device = self._hooks.resolve_device() if device is None else torch.device(device)

        for task_id, task_params in self._saved_params.items():
            task_importances = self._importances.get(task_id, {})
            for name, reference in task_params.items():
                if name not in task_importances:
                    continue

                importance = task_importances[name].to(target_device)
                reference = reference.to(target_device, dtype=importance.dtype)
                if name not in self._penalty_importance_sum:
                    self._penalty_importance_sum[name] = torch.zeros_like(importance)
                    self._penalty_reference_sum[name] = torch.zeros_like(reference)
                    self._penalty_reference_square_sum[name] = torch.zeros_like(reference)

                self._penalty_importance_sum[name] += importance
                self._penalty_reference_sum[name] += importance * reference
                self._penalty_reference_square_sum[name] += importance * reference.pow(2)

    def _get_current_params(self) -> Dict[str, torch.Tensor]:
        module = self._hooks.resolve_module()
        if module is None:
            return {}

        return {name: param.detach().cpu().clone() for name, param in module.named_parameters() if param.requires_grad}

    def _keep_latest_task_only(self) -> None:
        if not self._saved_params or not self._importances:
            return

        latest_task = max(self._saved_params)
        self._saved_params = {latest_task: self._saved_params[latest_task]}
        self._importances = {latest_task: self._importances[latest_task]}
        self._rebuild_penalty_cache()
