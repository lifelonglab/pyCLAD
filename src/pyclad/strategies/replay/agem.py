import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer
from pyclad.strategies.strategy import ConceptAgnosticStrategy, ConceptAwareStrategy, ConceptIncrementalStrategy


class AGEMStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    """A-GEM strategy using a reservoir replay buffer.

    Args:
        model: Torch-backed model whose parameters are trained directly.
        buffer: Reservoir buffer storing samples from previously seen concepts.
        batch_size: Training batch size for the current concept.
        replay_batch_size: Number of replay samples used to estimate the reference gradient.
            Defaults to the current mini-batch size when not provided.
        epochs: Number of training epochs per concept.
        projection_tolerance: Tolerance around zero when deciding if gradients conflict.
        device: Device to move input batches and replay samples to before training.
    """

    def __init__(
        self,
        model: TorchBackbone,
        buffer: ReservoirBuffer,
        batch_size: int = 32,
        replay_batch_size: int | None = None,
        epochs: int = 20,
        projection_tolerance: float = 1e-6,
        device: str | torch.device = "cpu",
    ) -> None:
        self._model = model
        self._buffer = buffer
        self._batch_size = batch_size
        self._replay_batch_size = replay_batch_size
        self._epochs = epochs
        self._projection_tolerance = projection_tolerance
        self._device = torch.device(device)
        self._current_batch_size = batch_size

    def learn(self, data: np.ndarray, concept_id: str | None = None) -> None:
        current = np.asarray(data, dtype=np.float32)
        if len(current) == 0:
            return

        loader = DataLoader(TensorDataset(torch.tensor(current, dtype=torch.float32)), self._batch_size, shuffle=True)
        self._model.fit_with_loss(loader, self._compute_loss, self._epochs, grad_callback=self._project_gradient)
        self._update_buffer(current)

    def _compute_loss(self, batch) -> Tensor:
        (x,) = batch
        x = x.to(self._device)
        self._current_batch_size = x.shape[0]
        return self._model.compute_loss(x)

    def _project_gradient(self, module: nn.Module) -> None:
        if len(self._buffer) == 0:
            return

        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        current_grad = self._capture_gradients(parameters)
        if not torch.isfinite(current_grad).all():
            raise FloatingPointError("A-GEM current gradients contain non-finite values.")

        module.zero_grad()
        replay_n = self._replay_batch_size or self._current_batch_size
        replay_x, _, _ = self._buffer.sample(n=replay_n, target_device=self._device)
        self._model.compute_loss(replay_x).backward()
        replay_grad = self._capture_gradients(parameters)
        if not torch.isfinite(replay_grad).all():
            raise FloatingPointError("A-GEM replay gradients contain non-finite values.")

        final_grad = self._project_if_needed(current_grad, replay_grad)
        module.zero_grad()
        self._restore_gradients(parameters, final_grad)

    def _project_if_needed(self, current: Tensor, replay: Tensor) -> Tensor:
        dot_product = torch.dot(current, replay)
        if dot_product.item() >= -self._projection_tolerance:
            return current

        replay_norm = torch.dot(replay, replay).clamp_min(1e-12)
        return current - (dot_product / replay_norm) * replay

    def _update_buffer(self, data: np.ndarray) -> None:
        x = torch.tensor(data, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            outputs = self._model.forward(x)
        self._buffer.update(x.detach(), outputs.detach(), x.detach())

    @staticmethod
    def _capture_gradients(parameters: list[nn.Parameter]) -> Tensor:
        parts = [
            (
                parameter.grad.detach().reshape(-1).clone()
                if parameter.grad is not None
                else torch.zeros(parameter.numel(), device=parameter.device, dtype=parameter.dtype)
            )
            for parameter in parameters
        ]
        return torch.cat(parts) if parts else torch.empty(0)

    @staticmethod
    def _restore_gradients(parameters: list[nn.Parameter], gradient: Tensor) -> None:
        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            parameter.grad = gradient[offset : offset + numel].view_as(parameter).clone()
            offset += numel

    def predict(self, data: np.ndarray, concept_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "AGEM"

    def additional_info(self) -> dict:
        return {
            "model": self._model.name(),
            "batch_size": self._batch_size,
            "replay_batch_size": self._replay_batch_size or self._batch_size,
            "epochs": self._epochs,
            "projection_tolerance": self._projection_tolerance,
            "device": str(self._device),
            "buffer": self._buffer.info(),
        }
