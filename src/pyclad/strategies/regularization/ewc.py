from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class EWCStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """
    Elastic Weight Consolidation (online variant).

    After each concept, approximates the diagonal Fisher Information Matrix via squared gradients of the
    reconstruction loss. When learning a new concept, adds a quadratic penalty that slows down changes to parameters
    that were important for previous concepts. Fisher is accumulated across all seen concepts.

    Larger fisher_batch_size underestimates the Fisher because it squares the mean gradient instead of averaging
    squared gradients. This shrinks the penalty and weakens protection against forgetting.
    """

    def __init__(
        self,
        model: TorchBackbone,
        lambda_ewc: float = 1.0,
        epochs: int = 20,
        batch_size: int = 32,
        fisher_batch_size: int = 1,
    ):
        self._model = model
        self._lambda = lambda_ewc
        self._epochs = epochs
        self._batch_size = batch_size
        self._fisher_batch_size = fisher_batch_size

        self._fisher: Dict[str, Tensor] = {}
        self._optimal_params: Dict[str, Tensor] = {}

    def learn(self, data: np.ndarray, **kwargs) -> None:
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        self._model.fit_with_loss(dataloader, self._compute_loss, self._epochs)
        self._update_fisher(data)

    def predict(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "EWC"

    def additional_info(self) -> Dict[str, Any]:
        return {"lambda_ewc": self._lambda, "epochs": self._epochs}

    # ------------------------------------------------------------------

    def _compute_loss(self, batch) -> Tensor:
        (x,) = batch
        loss = self._model.compute_loss(x)
        if self._fisher:
            loss = loss + self._lambda * self._penalty(self._model.get_module())
        return loss

    def _penalty(self, module: nn.Module) -> Tensor:
        penalty = torch.zeros((), device=next(module.parameters()).device)
        for name, param in module.named_parameters():
            if name in self._fisher:
                penalty = penalty + (self._fisher[name] * (param - self._optimal_params[name]).pow(2)).sum()
        return penalty

    def _update_fisher(self, data: np.ndarray) -> None:
        module = self._model.get_module()
        module.eval()

        loader = DataLoader(
            TensorDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=self._fisher_batch_size,
            shuffle=False,
        )

        new_fisher = {name: torch.zeros_like(param) for name, param in module.named_parameters()}
        n_samples = 0

        for (x,) in loader:
            loss = self._model.compute_loss(x)
            module.zero_grad()
            loss.backward()
            for name, param in module.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.detach().pow(2) * x.shape[0]
            n_samples += x.shape[0]

        for name in new_fisher:
            new_fisher[name] /= n_samples
            if name in self._fisher:
                self._fisher[name] += new_fisher[name]
            else:
                self._fisher[name] = new_fisher[name]

        self._optimal_params = {name: param.data.clone() for name, param in module.named_parameters()}
        module.train()
