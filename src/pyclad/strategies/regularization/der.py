import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.torch_backed_model import TorchBackedModel
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
        model: TorchBackedModel,
        buffer: ReservoirBuffer,
        alpha: float = 0.5,
        beta: float = 0.5,
        batch_size: int = 32,
        lr: float = 1e-2,
        epochs: int = 20,
    ) -> None:
        """
        :param model: torch-backed model whose parameters are trained directly.
        :param buffer: reservoir buffer storing past (input, output) pairs.
        :param alpha: weight of the output-consolidation term.
        :param beta: weight of the replay term. Setting beta=0
            reduces the strategy to plain DER.
        :param batch_size: training batch size.
        :param lr: learning rate for the internal Adam optimizer.
        :param epochs: number of training epochs per concept.
        """
        self._model = model
        self._buffer = buffer
        self._alpha = alpha
        self._beta = beta
        self._batch_size = batch_size
        self._lr = lr
        self._epochs = epochs

    def learn(self, data: np.ndarray) -> None:
        module = self._model.trainable_module
        device = next(module.parameters()).device

        module.train()
        optimizer = torch.optim.Adam(module.parameters(), lr=self._lr)

        loader = DataLoader(
            TensorDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=self._batch_size,
            shuffle=True,
        )

        for _ in range(self._epochs):
            for (x,) in loader:
                x = x.to(device)
                optimizer.zero_grad()

                step = module.training_step((x,), 0)
                loss = step["loss"]

                if len(self._buffer) > 0:
                    n = x.shape[0]
                    x_alpha, z_alpha, _ = self._buffer.sample(n=n, target_device=device)
                    x_beta, _, _ = self._buffer.sample(n=n, target_device=device)
                    alpha_step = module.training_step((x_alpha,), 0)
                    beta_step = module.training_step((x_beta,), 0)
                    loss = loss + self._alpha * F.mse_loss(alpha_step["output"], z_alpha)
                    loss = loss + self._beta * beta_step["loss"]

                loss.backward()
                optimizer.step()
                self._buffer.update(
                    step["input"].detach(),
                    step["output"].detach(),
                    step["target"].detach(),
                )

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "DER++"

    def additional_info(self) -> dict:
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "batch_size": self._batch_size,
            "lr": self._lr,
            "epochs": self._epochs,
            "buffer": self._buffer.info(),
        }
