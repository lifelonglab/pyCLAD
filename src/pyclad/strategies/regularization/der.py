import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.models.training.loaders import float_tensor_loader
from pyclad.models.training.runners.runner import TorchRunner
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer
from pyclad.strategies.strategy import ConceptAgnosticStrategy


class DerPlusPlus(ConceptAgnosticStrategy):
    """Strategy including DER++.

    Maintains a reservoir-sampled memory of past (input, output) pairs.
    The strategy drives its own training loop so it can augment the
    base loss per batch with two regularization terms:

    - alpha - responsible for preservation of past outputs.
    - beta - additional protection against distribution shift via replay.

    See: https://arxiv.org/abs/2004.07211
    """

    def __init__(
        self,
        *,
        model: TorchBackbone,
        runner: TorchRunner,
        buffer: ReservoirBuffer,
        alpha: float = 0.5,
        beta: float = 0.5,
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        :param model: torch-backed model whose parameters are trained directly.
            The optimizer and learning rate are managed by the model.
        :param runner: training-loop runner (plain or early-stopping).
        :param buffer: reservoir buffer storing past (input, output) pairs.
        :param alpha: weight of the output-consolidation term.
        :param beta: weight of the replay term. Setting beta=0
            reduces the strategy to plain DER.
        :param batch_size: training batch size.
        :param device: device to move input batches to before training.
        """
        self._model = model
        self._runner = runner
        self._buffer = buffer
        self._alpha = alpha
        self._beta = beta
        self._batch_size = batch_size
        self._device = torch.device(device)

    def learn(self, data: np.ndarray) -> None:
        train, val = self._runner.split_train_test(data)
        self._runner.run(
            self._model,
            float_tensor_loader(train, self._batch_size, shuffle=True),
            self._compute_loss,
            val_loader=float_tensor_loader(val, self._batch_size, shuffle=False) if val is not None else None,
            val_loss_fn=lambda batch: self._model.compute_loss(batch[0].to(self._device)),
        )

    def _compute_loss(self, batch) -> Tensor:
        (x,) = batch
        x = x.to(self._device)

        with torch.no_grad():
            buf_output = self._model.forward(x)

        loss = self._model.compute_loss(x)

        if len(self._buffer) > 0:
            n = x.shape[0]
            x_alpha, z_alpha, _ = self._buffer.sample(n=n, target_device=self._device)
            x_beta, _, _ = self._buffer.sample(n=n, target_device=self._device)
            loss = loss + self._alpha * F.mse_loss(self._model.forward(x_alpha), z_alpha)
            loss = loss + self._beta * self._model.compute_loss(x_beta)

        self._buffer.update(x.detach(), buf_output, x.detach())
        return loss

    def predict(self, data: np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "DER++"

    def additional_info(self) -> dict:
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "batch_size": self._batch_size,
            "runner": self._runner.info(),
            "device": str(self._device),
            "buffer": self._buffer.info(),
        }
