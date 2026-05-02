import logging
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.neural_model import NeuralTrainableModel
from pyclad.strategies.neural_hooks import BatchLoss, DataTransform, NeuralStrategyHooks
from pyclad.strategies.replay.buffers.balanced import BalancedReplayBuffer
from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.replay.replay import ReplayOnlyStrategy

logger = logging.getLogger(__name__)


class AGEMStrategy(ReplayOnlyStrategy):
    """A-GEM replay strategy for torch-backed models.

    The strategy projects the current gradient away from replay gradients when
    they conflict. SGD preserves the usual A-GEM update direction; Adam remains
    available as a heuristic.
    """

    def __init__(
        self,
        model: NeuralTrainableModel,
        buffer: Optional[ReplayBuffer] = None,
        *,
        module: Optional[torch.nn.Module] = None,
        loss_fn: Optional[BatchLoss] = None,
        data_transform: Optional[DataTransform] = None,
        batch_size: int = 32,
        replay_batch_size: Optional[int] = None,
        lr: float = 1e-2,
        optimizer: str = "sgd",
        projection_tolerance: float = 1e-6,
        epochs: int = 20,
        device: Union[torch.device, str, None] = None,
        shuffle: bool = True,
        buffer_size: int = 500,
        seed: int = 7,
    ):
        if module is not None and not isinstance(module, torch.nn.Module):
            raise TypeError("AGEMStrategy requires 'module' to be a torch.nn.Module.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if replay_batch_size is not None and replay_batch_size <= 0:
            raise ValueError(f"replay_batch_size must be positive, got {replay_batch_size}")
        if projection_tolerance < 0:
            raise ValueError(f"projection_tolerance must be non-negative, got {projection_tolerance}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        optimizer_name = optimizer.lower()
        if optimizer_name not in {"sgd", "adam"}:
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer}")

        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._task_count = 0
        self._concept_to_index: Dict[str, int] = {}
        self._module = module
        self._explicit_module = module is not None
        self._loss_fn = loss_fn
        self._data_transform = NeuralStrategyHooks.identity_transform if data_transform is None else data_transform
        self._hooks = NeuralStrategyHooks(model, module=module, data_transform=self._data_transform)
        self._hooks.validate_trainable_model("AGEMStrategy")
        self._batch_size = int(batch_size)
        self._replay_batch_size = int(replay_batch_size) if replay_batch_size is not None else self._batch_size
        self._lr = float(lr)
        self._optimizer_name = optimizer_name
        self._projection_tolerance = float(projection_tolerance)
        self._epochs = int(epochs)
        self._device = self._hooks.resolve_device(explicit_device=device)
        self._shuffle = bool(shuffle)
        self._adam_warning_emitted = False

        replay_buffer = buffer if buffer is not None else BalancedReplayBuffer(max_size=buffer_size, seed=self._seed)
        super().__init__(model, replay_buffer)

    def learn(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> None:
        del kwargs

        current_data = np.asarray(data, dtype=np.float32)
        if len(current_data) == 0:
            return

        concept_name = concept_id or f"concept_{self._task_count}"
        concept_index = self._concept_index(concept_name)

        logger.info("AGEM learn on %s with %s samples", concept_name, len(current_data))
        replay_data_for_calibration = np.array(self._buffer.data(), copy=True)
        self._fit(current_data, replay_data_for_calibration)

        if isinstance(self._buffer, BalancedReplayBuffer):
            self._buffer.add(
                examples=current_data,
                concept_indices=np.full(len(current_data), concept_index, dtype=np.int64),
            )
        else:
            self._buffer.update(current_data)

        calibration_data = current_data
        if len(replay_data_for_calibration) > 0:
            calibration_data = np.concatenate([replay_data_for_calibration, current_data], axis=0)

        self._hooks.after_fit(calibration_data)

        self._task_count += 1

    def predict(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        del concept_id, kwargs
        return self._hooks.predict(data)

    def name(self) -> str:
        return "AGEM"

    def additional_info(self) -> Dict:
        info = super().additional_info()
        info.update(
            {
                "model": self._model_name(),
                "seed": self._seed,
                "task_count": self._task_count,
                "concept_count": len(self._concept_to_index),
                "batch_size": self._batch_size,
                "replay_batch_size": self._replay_batch_size,
                "lr": self._lr,
                "optimizer": self._optimizer_name,
                "projection_tolerance": self._projection_tolerance,
                "epochs": self._epochs,
                "device": str(self._device),
                "shuffle": self._shuffle,
            }
        )
        return info

    def _fit(self, current_data: np.ndarray, replay_data: Optional[np.ndarray] = None) -> None:
        replay_data = np.asarray(replay_data, dtype=np.float32) if replay_data is not None else np.empty((0,))
        self._prepare_fit(current_data, replay_data)
        tensor_data = self._prepare_data(current_data)
        drop_last = self._drop_last()
        dataloader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=drop_last,
            num_workers=0,
        )
        if len(dataloader) == 0:
            logger.warning("Skipping AGEM fit because no current batches are available after batching.")
            return

        module = self._resolve_module()
        device = self._device
        module.to(device)
        module.train()

        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise TypeError("AGEMStrategy requires at least one trainable parameter.")

        if self._optimizer_name == "adam":
            if not self._adam_warning_emitted:
                logger.warning(
                    "AGEM with Adam does not preserve the standard A-GEM guarantee because Adam rescales "
                    "projected gradients with adaptive moment estimates. Use optimizer='sgd' for "
                    "guarantee-preserving updates."
                )
                self._adam_warning_emitted = True
            optimizer = torch.optim.Adam(parameters, lr=self._lr)
        else:
            optimizer = torch.optim.SGD(parameters, lr=self._lr)
        epochs = self._epochs
        has_replay_data = len(replay_data) > 0

        for epoch in range(epochs):
            epoch_current = 0.0
            epoch_replay = 0.0
            projection_count = 0

            for (batch,) in dataloader:
                batch = batch.to(device)

                optimizer.zero_grad()
                current_loss = self._batch_loss(module, batch)
                current_loss.backward()
                current_grad = self.capture_gradients(parameters)
                if not self.gradients_are_finite(current_grad):
                    logger.warning("Skipping AGEM update because the current gradient contains non-finite values.")
                    optimizer.zero_grad()
                    continue
                final_grad = current_grad

                replay_loss = torch.tensor(0.0, device=device)
                if has_replay_data:
                    replay_examples = self._sample_replay_examples(self._replay_batch_size)
                    if replay_examples is not None:
                        if drop_last and len(replay_examples) < 2:
                            optimizer.zero_grad()
                            self.restore_gradients(parameters, final_grad)
                            optimizer.step()
                            epoch_current += current_loss.item()
                            continue
                        optimizer.zero_grad()
                        replay_batch = self._prepare_data(replay_examples).to(device)
                        replay_loss = self._batch_loss(module, replay_batch)
                        replay_loss.backward()
                        replay_grad = self.capture_gradients(parameters)
                        if not self.gradients_are_finite(replay_grad):
                            logger.warning(
                                "Skipping AGEM update because the replay gradient contains non-finite values."
                            )
                            optimizer.zero_grad()
                            continue

                        if self.should_project(current_grad, replay_grad, self._projection_tolerance):
                            final_grad = self.project_gradient(current_grad, replay_grad)
                            projection_count += 1

                optimizer.zero_grad()
                if not self.gradients_are_finite(final_grad):
                    logger.warning("Skipping AGEM update because the final AGEM gradient contains non-finite values.")
                    continue
                self.restore_gradients(parameters, final_grad)
                optimizer.step()

                epoch_current += current_loss.item()
                epoch_replay += replay_loss.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                n_batches = len(dataloader)
                projection_rate = 100.0 * projection_count / n_batches
                logger.info(
                    "AGEM epoch %s/%s: current=%.6f replay=%.6f projections=%s/%s (%.1f%%)",
                    epoch + 1,
                    epochs,
                    epoch_current / n_batches,
                    epoch_replay / n_batches,
                    projection_count,
                    n_batches,
                    projection_rate,
                )

    def _sample_replay_examples(self, batch_size: int) -> Optional[np.ndarray]:
        if isinstance(self._buffer, BalancedReplayBuffer):
            if self._buffer.is_empty():
                return None
            if batch_size >= len(self._buffer):
                return self._buffer.data()
            return self._buffer.sample(batch_size)["examples"]

        replay_data = self._buffer.data()
        if len(replay_data) == 0:
            return None

        sample_size = min(int(batch_size), len(replay_data))
        if sample_size == len(replay_data):
            return np.array(replay_data, copy=True)
        indices = self._rng.choice(len(replay_data), size=sample_size, replace=False)
        return np.asarray(replay_data)[indices]

    def _concept_index(self, concept_id: str) -> int:
        if concept_id not in self._concept_to_index:
            self._concept_to_index[concept_id] = len(self._concept_to_index)
        return self._concept_to_index[concept_id]

    def _prepare_fit(self, current_data: np.ndarray, replay_data: np.ndarray) -> None:
        self._hooks.prepare_fit(current_data, replay_data=replay_data)

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        return self._hooks.prepare_data(data)

    def _resolve_module(self) -> torch.nn.Module:
        module = self._hooks.resolve_module()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("AGEMStrategy requires a torch.nn.Module via 'module' or model.module_getter().")
        self._module = module
        return module

    def _batch_loss(self, module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return self._hooks.batch_loss(module, batch, explicit_loss_fn=self._loss_fn)

    def _drop_last(self) -> bool:
        return self._hooks.drop_last()

    def _model_name(self) -> str:
        name = getattr(self._model, "name", None)
        return name() if callable(name) else self._model.__class__.__name__

    @staticmethod
    def capture_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        devices = {parameter.device for parameter in parameters}
        if len(devices) > 1:
            raise ValueError(
                "AGEMStrategy requires all trainable parameters to live on a single device; "
                "model-parallel parameter sets are not supported."
            )

        flat_parts = []
        for parameter in parameters:
            if parameter.grad is None:
                flat_parts.append(torch.zeros(parameter.numel(), device=parameter.device, dtype=parameter.dtype))
            else:
                flat_parts.append(parameter.grad.detach().reshape(-1).clone())

        if not flat_parts:
            return torch.empty(0)

        return torch.cat(flat_parts)

    @staticmethod
    def restore_gradients(parameters: Sequence[torch.nn.Parameter], flat_gradient: torch.Tensor) -> None:
        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            grad_slice = flat_gradient[offset : offset + numel].view_as(parameter)
            parameter.grad = grad_slice.clone()
            offset += numel

    @staticmethod
    def project_gradient(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if current.numel() == 0 or reference.numel() == 0:
            return current

        reference_norm = torch.dot(reference, reference)
        if not torch.isfinite(reference_norm) or reference_norm.item() <= 0:
            return current

        correction_numerator = torch.dot(current, reference)
        if not torch.isfinite(correction_numerator):
            return current

        correction = correction_numerator / reference_norm
        if not torch.isfinite(correction):
            return current

        return current - correction * reference

    @staticmethod
    def should_project(current: torch.Tensor, reference: torch.Tensor, tolerance: float = 0.0) -> bool:
        if current.numel() == 0 or reference.numel() == 0:
            return False

        dot_product = torch.dot(current, reference)
        if not torch.isfinite(dot_product):
            logger.warning("Skipping AGEM projection because the gradient dot product is non-finite.")
            return False

        return dot_product.item() < -float(tolerance)

    @staticmethod
    def gradients_are_finite(gradient: torch.Tensor) -> bool:
        return bool(torch.isfinite(gradient).all())
