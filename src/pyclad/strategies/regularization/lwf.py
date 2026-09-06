import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.models.training.loaders import float_tensor_loader
from pyclad.models.training.runners.runner import TorchRunner
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptIncrementalStrategy,
)


class LwFStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    """Learning without Forgetting using output distillation."""

    def __init__(
        self,
        model: TorchBackbone,
        runner: TorchRunner,
        alpha: float = 0.5,
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        :param model: torch-backed model whose parameters are trained directly.
        :param runner: training-loop runner (plain or early-stopping).
        :param alpha: weight of the teacher-output distillation term.
        :param batch_size: training batch size.
        :param device: device to move input batches to before training.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._model = model
        self._runner = runner
        self._alpha = alpha
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._task_count = 0
        self._teacher: TorchBackbone | None = None

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        del args, kwargs
        train, val = self._runner.split_train_test(data)
        self._runner.run(
            self._model,
            float_tensor_loader(train, self._batch_size, shuffle=True),
            self._compute_loss,
            val_loader=float_tensor_loader(val, self._batch_size, shuffle=False) if val is not None else None,
            val_loss_fn=lambda batch: self._model.compute_loss(batch[0].to(self._device)),
        )
        self._task_count += 1
        self._teacher = self._clone_teacher()

    def _compute_loss(self, batch) -> Tensor:
        (x,) = batch
        x = x.to(self._device)
        loss = self._model.compute_loss(x)

        if self._teacher is not None and self._alpha != 0:
            with torch.no_grad():
                teacher_output = self._teacher.forward(x)
            student_output = self._model.forward(x)
            loss = loss + self._alpha * F.mse_loss(student_output, teacher_output)

        return loss

    def _clone_teacher(self) -> TorchBackbone:
        teacher = copy.deepcopy(self._model)
        module = teacher.get_module()
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
        return teacher

    def predict(self, data: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        return self._model.predict(data)

    def name(self) -> str:
        return "LwF"

    def additional_info(self) -> dict:
        return {
            "model": self._model.name(),
            "alpha": self._alpha,
            "batch_size": self._batch_size,
            "runner": self._runner.info(),
            "device": str(self._device),
            "task_count": self._task_count,
            "has_teacher": self._teacher is not None,
        }
