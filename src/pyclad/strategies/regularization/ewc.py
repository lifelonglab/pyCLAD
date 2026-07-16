from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.models.training.loaders import float_tensor_loader
from pyclad.models.training.runners.runner import TorchRunner
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class EWCStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """
    Elastic Weight Consolidation with separate offline and online penalty modes.

    After each concept, approximates the diagonal Fisher Information Matrix via squared gradients of the
    reconstruction loss. When learning a new concept, adds a quadratic penalty that slows down changes to parameters
    that were important for previous concepts.

    By default, stores the contribution from every seen concept. With online=True, keeps the accumulated Fisher
    anchored to the most recent concept parameters, matching the lightweight latest-concept variant.

    Larger fisher_batch_size underestimates the Fisher because it squares the mean gradient instead of averaging
    squared gradients. This shrinks the penalty and weakens protection against forgetting.
    """

    def __init__(
        self,
        model: TorchBackbone,
        runner: TorchRunner,
        lambda_ewc: float = 1.0,
        batch_size: int = 32,
        fisher_batch_size: int = 1,
        online: bool = False,
    ):
        self._model = model
        self._runner = runner
        self._lambda = lambda_ewc
        self._batch_size = batch_size
        self._fisher_batch_size = fisher_batch_size
        self._online = online

        self._importance_sum: Dict[str, Tensor] = {}
        self._reference_sum: Dict[str, Tensor] = {}
        self._reference_square_sum: Dict[str, Tensor] = {}

    def learn(self, data: np.ndarray, **kwargs) -> None:
        train, val = self._runner.split_train_test(data)
        self._runner.run(
            self._model,
            float_tensor_loader(train, self._batch_size, shuffle=True),
            self._compute_loss,
            val_loader=float_tensor_loader(val, self._batch_size, shuffle=False) if val is not None else None,
            val_loss_fn=lambda batch: self._model.compute_loss(batch[0]),
        )
        self._update_fisher(train)

    def predict(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "EWC"

    def additional_info(self) -> Dict[str, Any]:
        return {"lambda_ewc": self._lambda, "online": self._online, "runner": self._runner.info()}

    # ------------------------------------------------------------------

    def _compute_loss(self, batch) -> Tensor:
        (x,) = batch
        loss = self._model.compute_loss(x)
        if self._importance_sum:
            loss = loss + self._lambda * self._penalty(self._model.get_module())
        return loss

    def _penalty(self, module: nn.Module) -> Tensor:
        penalty = torch.zeros((), device=next(module.parameters()).device)
        for name, param in module.named_parameters():
            if not param.requires_grad or name not in self._importance_sum:
                continue

            importance_sum = self._importance_sum[name].to(device=param.device, dtype=param.dtype)
            reference_sum = self._reference_sum[name].to(device=param.device, dtype=param.dtype)
            reference_square_sum = self._reference_square_sum[name].to(device=param.device, dtype=param.dtype)
            penalty = (
                penalty + (importance_sum * param.pow(2) - 2.0 * reference_sum * param + reference_square_sum).sum()
            )
        return penalty

    def _update_fisher(self, data: np.ndarray) -> None:
        module = self._model.get_module()
        module.eval()

        loader = float_tensor_loader(data, self._fisher_batch_size, shuffle=False)

        new_fisher = {name: torch.zeros_like(param) for name, param in module.named_parameters() if param.requires_grad}
        n_samples = 0

        for (x,) in loader:
            module.zero_grad()
            loss = self._model.compute_loss(x)
            loss.backward()
            for name, param in module.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.detach().pow(2) * x.shape[0]
            n_samples += x.shape[0]

        params = dict(module.named_parameters())
        for name in new_fisher:
            new_fisher[name] /= n_samples
            param = params[name].detach()
            reference = new_fisher[name] * param
            reference_square = new_fisher[name] * param.pow(2)
            if name in self._importance_sum:
                self._importance_sum[name] += new_fisher[name]
                if self._online:
                    self._reference_sum[name] = self._importance_sum[name] * param
                    self._reference_square_sum[name] = self._importance_sum[name] * param.pow(2)
                else:
                    self._reference_sum[name] += reference
                    self._reference_square_sum[name] += reference_square
            else:
                self._importance_sum[name] = new_fisher[name]
                self._reference_sum[name] = reference
                self._reference_square_sum[name] = reference_square

        module.train()
