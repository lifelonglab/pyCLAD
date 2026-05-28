from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

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
        self._column_score_stats: list[tuple[float, float]] = []
        self._concept_to_column: dict[str, int] = {}
        self._current_column = -1

    @property
    def current_column(self) -> int:
        return self._current_column

    @property
    def num_columns(self) -> int:
        return len(self._columns)

    def learn(self, data: np.ndarray, concept_id: str | None = None, **kwargs) -> None:
        if concept_id is None and not self._task_free:
            raise ValueError("PNNStrategy requires concept_id for learning unless task_free=True.")
        if concept_id is not None and concept_id in self._concept_to_column:
            raise ValueError(f"PNNStrategy has already learned concept_id '{concept_id}'.")

        column_index = self._add_column(concept_id)
        self._fit_column(column_index, data)
        self._freeze_column(column_index)
        self._current_column = column_index

    def _add_column(self, concept_id: str | None) -> int:
        model = self._base_model_factory()
        model.to(self._device)
        self._columns.append(model)
        self._column_score_stats.append((0.0, 1.0))
        column_index = len(self._columns) - 1
        if concept_id is not None:
            self._concept_to_column[concept_id] = column_index
        return column_index

    def _fit_column(self, column_index: int, data: np.ndarray) -> None:
        model = self._column(column_index)
        self._set_trainable(model, True)
        current = np.asarray(data, dtype=np.float32)
        loader = DataLoader(
            torch.tensor(current, dtype=torch.float32, device=self._device), self._batch_size, shuffle=True
        )

        model.fit_with_loss(loader, model.compute_loss, self._epochs)
        self._column_score_stats[column_index] = self._score_stats(model, current)

    @staticmethod
    def _score_stats(model: TorchBackbone, data: np.ndarray) -> tuple[float, float]:
        _, scores = model.predict(data)
        scores = np.asarray(scores, dtype=np.float32)
        std = float(scores.std())
        if std == 0.0:
            std = 1.0
        return float(scores.mean()), std

    def _freeze_column(self, column_index: int) -> None:
        self._set_trainable(self._column(column_index), False)

    @staticmethod
    def _set_trainable(model: TorchBackbone, trainable: bool) -> None:
        module = model.get_module()
        module.train(mode=trainable)
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _column(self, column_index: int) -> TorchBackbone:
        if column_index < 0 or column_index >= len(self._columns):
            raise ValueError(f"Invalid column index {column_index}.")
        return self._columns[column_index]

    def predict(
        self,
        data: np.ndarray,
        concept_id: str | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._columns:
            raise ValueError("PNNStrategy cannot predict before learning at least one concept.")

        if concept_id is not None:
            if concept_id not in self._concept_to_column:
                raise ValueError(f"Unknown concept_id '{concept_id}'.")
            return self._column(self._concept_to_column[concept_id]).predict(data)
        if self._task_free:
            return self._task_free_predict(data)

        raise ValueError("PNNStrategy requires concept_id for prediction unless task_free=True.")

    def _task_free_predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels, scores = zip(*(column.predict(data) for column in self._columns))
        labels = np.stack([np.asarray(label) for label in labels], axis=0)
        scores = np.stack([np.asarray(score) for score in scores], axis=0)
        stats = np.asarray(self._column_score_stats, dtype=np.float32)
        score_shape = (len(self._columns),) + (1,) * (scores.ndim - 1)
        normalized_scores = (scores - stats[:, 0].reshape(score_shape)) / stats[:, 1].reshape(score_shape)
        best = np.expand_dims(np.argmin(normalized_scores, axis=0), axis=0)
        return (
            np.take_along_axis(labels, best, axis=0).squeeze(axis=0),
            np.take_along_axis(normalized_scores, best, axis=0).squeeze(axis=0),
        )

    def name(self) -> str:
        return "PNN"

    def additional_info(self) -> dict:
        return {
            "task_free": self._task_free,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "device": str(self._device),
            "current_column": self._current_column,
            "num_columns": len(self._columns),
            "known_concepts": len(self._concept_to_column),
        }
