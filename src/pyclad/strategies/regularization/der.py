import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.torch_backbone import TorchBackbone
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
        buffer: ReservoirBuffer,
        alpha: float = 0.5,
        beta: float = 0.5,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        :param model: torch-backed model whose parameters are trained directly.
            The optimizer and learning rate are managed by the model.
        :param buffer: reservoir buffer storing past (input, output) pairs.
        :param alpha: weight of the output-consolidation term.
        :param beta: weight of the replay term. Setting beta=0
            reduces the strategy to plain DER.
        :param batch_size: training batch size.
        :param epochs: number of training epochs per concept.
        :param device: device to move input batches to before training.
        """
        self._model = model
        self._buffer = buffer
        self._alpha = alpha
        self._beta = beta
        self._batch_size = batch_size
        self._epochs = epochs
        self._device = torch.device(device)

    def learn(self, data: np.ndarray) -> None:
        loader = DataLoader(
            TensorDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=self._batch_size,
            shuffle=True,
        )
        self._model.fit_with_loss(loader, self._compute_loss, self._epochs)

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

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "DER++"

    def additional_info(self) -> dict:
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "device": str(self._device),
            "buffer": self._buffer.info(),
        }
