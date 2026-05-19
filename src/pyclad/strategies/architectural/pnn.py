from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class PNNStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Compact progressive-column strategy for torch backbones."""

    def __init__(
        self,
        base_model_factory: Callable[[], TorchBackbone],
        batch_size: int = 32,
        epochs: int = 20,
        task_free: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        self._base_model_factory = base_model_factory
        self._batch_size = batch_size
        self._epochs = epochs
        self._task_free = task_free
        self._device = torch.device(device)
        self._columns: list[TorchBackbone] = []
        self._concept_to_task: dict[str, int] = {}
        self._current_task = -1

    @property
    def current_task(self) -> int:
        return self._current_task

    @property
    def num_columns(self) -> int:
        return len(self._columns)

    def learn(self, data: np.ndarray, concept_id: str | None = None, **kwargs) -> None:
        del kwargs
        task_label = self._concept_to_task.get(concept_id) if concept_id is not None else None
        if task_label is None:
            task_label = self._add_column(concept_id)
        self._fit_column(task_label, data)
        self._freeze_column(task_label)
        self._current_task = task_label

    def _add_column(self, concept_id: str | None) -> int:
        model = self._base_model_factory()
        model.to(self._device)
        self._columns.append(model)
        task_label = len(self._columns) - 1
        if concept_id is not None:
            self._concept_to_task[concept_id] = task_label
        return task_label

    def _fit_column(self, task_label: int, data: np.ndarray) -> None:
        model = self._column(task_label)
        self._set_trainable(model, True)
        current = np.asarray(data, dtype=np.float32)
        loader = DataLoader(TensorDataset(torch.tensor(current, dtype=torch.float32)), self._batch_size, shuffle=True)

        def loss_fn(batch) -> torch.Tensor:
            (x,) = batch
            return model.compute_loss(x.to(self._device))

        model.fit_with_loss(loader, loss_fn, self._epochs)

    def _freeze_column(self, task_label: int) -> None:
        self._set_trainable(self._column(task_label), False)

    @staticmethod
    def _set_trainable(model: TorchBackbone, trainable: bool) -> None:
        module = model.get_module()
        module.train(mode=trainable)
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _column(self, task_label: int) -> TorchBackbone:
        if task_label < 0 or task_label >= len(self._columns):
            raise ValueError(f"Invalid task label {task_label}.")
        return self._columns[task_label]

    def predict(
        self,
        data: np.ndarray,
        task_label: int | None = None,
        concept_id: str | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        del kwargs
        if not self._columns:
            raise ValueError("PNNStrategy cannot predict before learning at least one concept.")

        if task_label is not None:
            return self._column(task_label).predict(data)
        if concept_id is not None and concept_id in self._concept_to_task:
            return self._column(self._concept_to_task[concept_id]).predict(data)
        if self._task_free:
            return self._task_free_predict(data)
        return self._column(self._current_task).predict(data)

    def _task_free_predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels, scores = zip(*(column.predict(data) for column in self._columns))
        labels = np.stack([np.asarray(label) for label in labels], axis=0)
        scores = np.stack([np.asarray(score) for score in scores], axis=0)
        best = np.argmin(scores, axis=0)
        sample_index = np.arange(scores.shape[1])
        return labels[best, sample_index], scores[best, sample_index]

    def name(self) -> str:
        return "PNN"

    def additional_info(self) -> dict:
        return {
            "task_free": self._task_free,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "device": str(self._device),
            "current_task": self._current_task,
            "num_columns": len(self._columns),
            "known_concepts": len(self._concept_to_task),
        }
