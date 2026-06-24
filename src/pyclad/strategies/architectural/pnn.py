from __future__ import annotations
from typing import Callable
import numpy as np, torch
import torch.nn.functional as F
from torch import nn; from torch.utils.data import DataLoader
from pyclad.models.torch_backbone import TorchBackbone
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy
class PNNStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    current_column = property(lambda self: len(self._columns) - 1); num_columns = property(lambda self: len(self._columns))
    def __init__(self, base_model_factory: Callable[[], TorchBackbone], batch_size: int = 32, epochs: int = 20, task_free: bool = False, device: str | torch.device = "cpu") -> None:
        self._base_model_factory = base_model_factory
        self._batch_size, self._epochs, self._task_free = batch_size, epochs, task_free
        self._device = torch.device(device)
        self._columns, self._adapters, self._score_stats = [], [], []
        self._concept_to_column = {}
    def learn(self, data: np.ndarray, concept_id: str | None = None, **kwargs) -> None:
        if concept_id is None and not self._task_free:
            raise ValueError("PNNStrategy requires concept_id for learning unless task_free=True.")
        if concept_id is not None and concept_id in self._concept_to_column:
            raise ValueError(f"PNNStrategy has already learned concept_id '{concept_id}'.")
        column = self._base_model_factory().to(self._device)
        module = column.get_module()
        if not hasattr(module, "encoder") or not hasattr(module, "decoder"):
            raise ValueError("PNNStrategy requires a backbone module with encoder and decoder attributes for hidden-layer lateral connections.")
        self._columns.append(column); self._adapters.append(nn.ModuleList([nn.ModuleList() for _ in self._layers(self.current_column)]).to(self._device))
        column_index = self.current_column
        if concept_id is not None:
            self._concept_to_column[concept_id] = column_index
        self._set_trainable(column_index, True); self._fit(column_index, data); self._set_trainable(column_index, False)
        self._score_stats.append(self._score_stats_for(column_index, data))
    def predict(self, data: np.ndarray, concept_id: str | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        if not self._columns:
            raise ValueError("PNNStrategy cannot predict before learning at least one concept.")
        if self._task_free:
            return self._predict_task_free(data)
        if concept_id is None:
            raise ValueError("PNNStrategy requires concept_id for prediction unless task_free=True.")
        return self._predict_column(self._concept_to_column[concept_id], data) if concept_id in self._concept_to_column else (np.zeros(data.shape[0]), np.zeros(data.shape[0]))
    def _fit(self, column_index: int, data: np.ndarray) -> None:
        loader = DataLoader(torch.tensor(data, dtype=torch.float32, device=self._device), self._batch_size, shuffle=True)
        optimizer = None; self._columns[column_index].get_module().train(); self._adapters[column_index].train()
        for _ in range(self._epochs):
            for batch in loader:
                loss = self._loss(column_index, batch); optimizer = optimizer or self._optimizer(column_index)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
    def _loss(self, column_index: int, x: torch.Tensor) -> torch.Tensor:
        return self._columns[column_index].compute_loss(x) if column_index == 0 else F.mse_loss(self._forward(column_index, x), x)
    def _optimizer(self, column_index: int) -> torch.optim.Optimizer:
        base = self._columns[column_index].get_optimizer()
        return type(base)((param for param in self._params(column_index) if param.requires_grad), **base.defaults)
    def _params(self, column_index: int):
        yield from self._columns[column_index].get_module().parameters()
        yield from self._adapters[column_index].parameters()
    def _set_trainable(self, column_index: int, trainable: bool) -> None:
        self._columns[column_index].get_module().train(mode=trainable); self._adapters[column_index].train(mode=trainable)
        for parameter in self._params(column_index):
            parameter.requires_grad = trainable
    def _score_stats_for(self, column_index: int, data: np.ndarray) -> tuple[float, float]:
        _, scores = self._predict_column(column_index, np.asarray(data, dtype=np.float32)); scores = np.asarray(scores, dtype=np.float32)
        return float(scores.mean()), float(scores.std()) or 1.0
    def _predict_task_free(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels, scores = zip(*(self._predict_column(index, data) for index in range(len(self._columns))))
        labels, scores = np.stack([np.asarray(label) for label in labels]), np.stack([np.asarray(score) for score in scores])
        stats = np.asarray(self._score_stats, dtype=np.float32); shape = (len(self._columns),) + (1,) * (scores.ndim - 1)
        scores = (scores - stats[:, 0].reshape(shape)) / stats[:, 1].reshape(shape)
        best = np.expand_dims(np.argmin(scores, axis=0), axis=0)
        return np.take_along_axis(labels, best, axis=0).squeeze(axis=0), np.take_along_axis(scores, best, axis=0).squeeze(axis=0)
    def _predict_column(self, column_index: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if column_index == 0:
            return self._columns[column_index].predict(data)
        data = np.asarray(data, dtype=np.float32)
        with torch.no_grad():
            x_hat = self._forward(column_index, torch.tensor(data, dtype=torch.float32, device=self._device)).cpu().numpy()
        scores = ((data - x_hat) ** 2).mean(axis=1)
        return (scores > getattr(self._columns[column_index], "threshold", 0.5)).astype(int), scores
    def _layers(self, column_index: int) -> list[nn.Module]:
        module = self._columns[column_index].get_module(); layers = []
        for block in (module.encoder, module.decoder):
            layers.extend(list(block) if isinstance(block, nn.Sequential) else [block])
        return layers
    def _forward(self, column_index: int, x: torch.Tensor, stop: int | None = None, cache=None) -> torch.Tensor:
        cache = {} if cache is None else cache; key = (column_index, stop)
        if key in cache: return cache[key]
        h = x
        for layer_index, layer in enumerate(self._layers(column_index)[: stop + 1 if stop is not None else None]):
            h = layer(h)
            if column_index:
                with torch.no_grad():
                    old = [self._forward(index, x, layer_index, cache).detach() for index in range(column_index)]
                adapters = self._adapters[column_index][layer_index]
                while len(adapters) < len(old):
                    adapter = nn.Linear(old[len(adapters)].flatten(1).shape[1], h.flatten(1).shape[1], bias=False).to(h.device)
                    adapter.requires_grad_(adapters.training); adapters.append(adapter)
                for adapter, old_activation in zip(adapters, old):
                    h = h + adapter(old_activation.flatten(1)).reshape_as(h)
        cache[key] = h; return h
    def name(self) -> str:
        return "PNN"
    def additional_info(self) -> dict:
        return {"task_free": self._task_free, "batch_size": self._batch_size, "epochs": self._epochs, "device": str(self._device), "current_column": self.current_column, "num_columns": len(self._columns), "known_concepts": len(self._concept_to_column)}
