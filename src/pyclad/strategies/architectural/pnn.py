from contextlib import contextmanager
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from pyclad.models.autoencoder.autoencoder import Autoencoder, AutoencoderModule
from pyclad.models.torch_backbone import TorchBackbone
from pyclad.models.training.loaders import float_tensor_loader
from pyclad.models.training.runners.runner import TorchRunner
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class _PNNAdapter(nn.Module):
    """Linear laterals (Eq. 1) or gated MLP laterals (Eq. 2) from Rusu et al."""

    def __init__(self, input_sizes: list[int], output_size: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.projections = nn.ModuleList([nn.Linear(size, output_size) for size in input_sizes])
        else:
            self.scales = nn.Parameter(torch.randn(len(input_sizes)))
            self.projection = nn.Sequential(
                nn.Linear(sum(input_sizes), output_size), nn.ReLU(), nn.Linear(output_size, output_size)
            )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        if self.kind == "linear":
            return sum(projection(value) for projection, value in zip(self.projections, inputs))
        scaled = [scale * value for scale, value in zip(self.scales, inputs)]
        return self.projection(torch.cat(scaled, dim=1))


class PNNStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Progressive columns with backbone-defined losses and anomaly scores.

    Dense autoencoder columns use matching transform/activation stages. Lateral inputs
    come from the preceding depth and are combined before activation, following
    Rusu et al. (2016), https://arxiv.org/abs/1606.04671.
    Task-free prediction is a pyCLAD extension that selects the smallest
    training-standardized score and returns it with the selected column's label.
    """

    def __init__(
        self,
        base_model_factory: Callable[[], TorchBackbone],
        batch_size: int = 32,
        epochs: int = 20,
        task_free: bool = False,
        device: str | torch.device = "cpu",
        adapter: str = "mlp",
        runner: TorchRunner | None = None,
    ) -> None:
        """
        :param base_model_factory: factory returning an independent dense autoencoder backbone for each concept.
            The backbone retains its loss, prediction, and optimizer methods.
        :param batch_size: training batch size.
        :param epochs: number of training epochs for each new column.
        :param task_free: select a column using standardized anomaly scores during prediction.
            Otherwise, concept identifiers are required for learning and prediction.
        :param device: device used for column training and prediction.
        :param runner: untrained training-loop runner template, copied for each column to isolate run state;
            defaults to StandardRunner configured with epochs.
        :param adapter: "mlp" for gated nonlinear adapters, or "linear" for additive projections.
        """
        if batch_size <= 0 or epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")

        if adapter not in {"linear", "mlp"}:
            raise ValueError("adapter must be 'linear' or 'mlp'.")

        self._runner = deepcopy(runner) if runner is not None else StandardRunner(max_epochs=epochs)
        self._adapter = adapter
        self._base_model_factory = base_model_factory
        self._batch_size = batch_size
        self._epochs = epochs
        self._task_free = task_free
        self._device = torch.device(device)
        self._columns = []
        self._adapters = []
        self._score_stats = []
        self._concept_to_column = {}

    @property
    def current_column(self) -> int:
        """Index of the latest column, or -1 before learning."""
        return len(self._columns) - 1

    @property
    def num_columns(self) -> int:
        """Number of columns currently stored."""
        return len(self._columns)

    def learn(self, data: np.ndarray, concept_id: str | None = None, **kwargs) -> None:
        """Train and calibrate a new column, keeping previous columns frozen."""
        if concept_id is None and not self._task_free:
            raise ValueError("PNNStrategy requires concept_id for learning unless task_free=True.")
        if concept_id is not None and concept_id in self._concept_to_column:
            raise ValueError(f"PNNStrategy has already learned concept_id '{concept_id}'.")

        data = self._validate_data(data)
        column = self._base_model_factory()
        if not isinstance(column, TorchBackbone):
            raise ValueError("PNNStrategy requires a TorchBackbone.")

        column_state = self._state_ids(column)
        if any(column_state & self._state_ids(old) for old in self._columns):
            raise ValueError("PNNStrategy requires independent columns from base_model_factory.")

        column.to(self._device)
        layers = self._layers(column)
        if not layers or any(len(self._layers(old)) != len(layers) for old in self._columns):
            raise ValueError("PNNStrategy requires matching, nonempty layer sequences.")

        self._columns.append(column)
        self._adapters.append(nn.ModuleList().to(self._device))
        column_index = self.current_column

        try:
            # Materialize adapters without updating dropout or batch-normalization state.
            self._set_trainable(column_index, False)
            with torch.no_grad():
                self._forward(column_index, torch.tensor(data[:1], device=self._device))

            for (transform, _), adapter in zip(layers, self._adapters[column_index]):
                transform.add_module("_pnn_lateral", adapter)
            self._set_trainable(column_index, True)
            self._fit(column_index, data)
            self._set_trainable(column_index, False)

            _, scores = self._predict_column(column_index, data)
            scores = scores.astype(np.float64)
            mean = float(scores.mean())
            std = float(scores.std())
            if std == 0.0:
                std = 1.0
            stats = (mean, std)
            if not np.isfinite(stats).all():
                raise ValueError("PNNStrategy requires finite score statistics.")
        except Exception:
            self._set_trainable(column_index, False)
            for transform, _ in layers:
                if "_pnn_lateral" in transform._modules:
                    del transform._modules["_pnn_lateral"]
            self._columns.pop()
            self._adapters.pop()
            raise

        self._score_stats.append(stats)
        if concept_id is not None:
            self._concept_to_column[concept_id] = column_index

    def predict(self, data: np.ndarray, concept_id: str | None = None, **kwargs) -> PredictionResults:
        """Predict with a known concept or route by standardized anomaly score."""
        if not self._columns:
            raise ValueError("PNNStrategy cannot predict before learning at least one concept.")

        data = self._validate_data(data)
        if self._task_free:
            predictions = [self._predict_column(index, data) for index in range(self.num_columns)]
            labels = np.stack([label for label, _ in predictions])
            scores = np.stack([score for _, score in predictions])
            mean, std = np.asarray(self._score_stats).T
            scores = (scores - mean[:, None]) / std[:, None]
            if not np.isfinite(scores).all():
                raise ValueError("PNNStrategy requires finite normalized scores.")

            best_columns = scores.argmin(axis=0)
            rows = np.arange(len(data))
            return PredictionResults(y_pred=labels[best_columns, rows], anomaly_scores=scores[best_columns, rows])

        if concept_id is None:
            raise ValueError("PNNStrategy requires concept_id for prediction unless task_free=True.")
        if concept_id not in self._concept_to_column:
            raise ValueError(f"Unknown concept_id '{concept_id}'.")

        column_index = self._concept_to_column[concept_id]
        labels, scores = self._predict_column(column_index, data)
        return PredictionResults(y_pred=labels, anomaly_scores=scores)

    def _fit(self, column_index: int, data: np.ndarray) -> None:
        column = self._columns[column_index]
        runner = deepcopy(self._runner)
        train, val = runner.split_train_test(data)

        def loss_fn(batch):
            loss = column.compute_loss(batch[0].to(self._device))
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("PNNStrategy requires a finite scalar loss.")
            return loss

        def check_gradients(module):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                raise ValueError("PNNStrategy requires finite gradients.")

        with self._lateral_connections(column_index):
            runner.run(
                column,
                float_tensor_loader(train, self._batch_size, shuffle=True),
                loss_fn,
                grad_callback=check_gradients,
                val_loader=float_tensor_loader(val, self._batch_size, shuffle=False) if val is not None else None,
                val_loss_fn=loss_fn,
            )
        if any(not torch.isfinite(p).all() for p in self._params(column_index)):
            raise ValueError("PNNStrategy requires finite parameters after training.")

    def _params(self, column_index: int):
        yield from self._columns[column_index].get_module().parameters()

    def _set_trainable(self, column_index: int, trainable: bool) -> None:
        self._columns[column_index].get_module().train(mode=trainable)
        self._adapters[column_index].train(mode=trainable)
        for parameter in self._params(column_index):
            parameter.requires_grad = trainable

    def _predict_column(self, column_index: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        column = self._columns[column_index]
        with torch.no_grad(), self._lateral_connections(column_index):
            if getattr(column.predict, "__func__", None) is Autoencoder.predict:
                # The built-in autoencoder scorer assumes CPU input and output.
                # Keep its reconstruction score and threshold on the requested device.
                column.get_module().eval()
                output = column.forward(torch.tensor(data, device=self._device)).cpu().numpy()
                scores = ((data - output) ** 2).mean(axis=1)
                result = PredictionResults(y_pred=(scores > column.threshold).astype(int), anomaly_scores=scores)
            else:
                result = column.predict(data)
        labels, scores = result.y_pred, result.anomaly_scores
        if labels.shape != (len(data),) or scores.shape != (len(data),) or not np.isfinite(scores).all():
            raise ValueError("PNNStrategy requires one label and finite anomaly score per sample.")
        return labels, scores

    @staticmethod
    def _state_ids(backbone: TorchBackbone) -> set[int]:
        """Identify modules and tensors that must not be shared between columns."""
        module = backbone.get_module()
        state = [*module.modules(), *module.parameters(), *module.buffers()]
        return {id(value) for value in state}

    @staticmethod
    def _validate_data(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or not data.size or not np.isfinite(data).all():
            raise ValueError("PNNStrategy requires nonempty, finite two-dimensional data.")
        return data

    def _forward(self, column_index: int, x: Tensor, return_activations: bool = False):
        activations = []
        for index in range(column_index + 1):
            h = x
            outputs = [x]
            train_column = torch.is_grad_enabled() and index == column_index
            with torch.set_grad_enabled(train_column):
                for depth, (transform, activation) in enumerate(self._layers(self._columns[index])):
                    h = transform(h)
                    if h.ndim != 2:
                        raise ValueError("PNNStrategy requires two-dimensional stage outputs.")
                    if index:
                        previous_inputs = [old[depth] for old in activations]
                        adapters = self._adapters[index]
                        if len(adapters) == depth:
                            input_sizes = [value.shape[1] for value in previous_inputs]
                            adapters.append(_PNNAdapter(input_sizes, h.shape[1], self._adapter).to(h))
                        h = h + adapters[depth](previous_inputs)
                    h = activation(h)
                    if h.ndim != 2:
                        raise ValueError("PNNStrategy requires two-dimensional stage outputs.")
                    outputs.append(h.clone())
            activations.append(outputs)
        return activations if return_activations else h

    @staticmethod
    def _layers(column: TorchBackbone):
        module = column.get_module()
        if not isinstance(module, AutoencoderModule):
            raise ValueError("PNNStrategy requires a dense AutoencoderModule.")

        def flatten(block):
            if isinstance(block, nn.Sequential):
                return [layer for child in block for layer in flatten(child)]
            return [block]

        stages = []
        following = []
        transform = None
        transforms = set()
        for layer in flatten(module.encoder) + flatten(module.decoder):
            if isinstance(layer, nn.Linear):
                if id(layer) in transforms:
                    raise ValueError("PNNStrategy requires distinct dense transform instances at each depth.")
                transforms.add(id(layer))
                if transform is not None:
                    stages.append((transform, nn.Sequential(*following)))
                transform, following = layer, []
            elif transform is not None and isinstance(
                layer,
                (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Identity, nn.Dropout, nn.BatchNorm1d),
            ):
                following.append(layer)
            else:
                raise ValueError("PNNStrategy requires explicit dense transform/activation boundaries.")
        if transform is not None:
            stages.append((transform, nn.Sequential(*following)))
        return stages

    @contextmanager
    def _lateral_connections(self, column_index):
        """Apply laterals while the model runs its existing loss or prediction path."""
        handles = []
        previous = []

        def prepare(module, inputs):
            with torch.no_grad():
                previous[:] = self._forward(column_index - 1, inputs[0], return_activations=True)

        def add_lateral(depth):
            def hook(module, inputs, output):
                return output + self._adapters[column_index][depth]([values[depth] for values in previous])

            return hook

        try:
            if column_index:
                module = self._columns[column_index].get_module()
                handles.append(module.register_forward_pre_hook(prepare))
                for depth, (transform, _) in enumerate(self._layers(self._columns[column_index])):
                    handles.append(transform.register_forward_hook(add_lateral(depth)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def name(self) -> str:
        return "PNN"

    def additional_info(self) -> dict:
        return {
            "runner": self._runner.info(),
            "adapter": self._adapter,
            "task_free": self._task_free,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "device": str(self._device),
            "current_column": self.current_column,
            "num_columns": len(self._columns),
            "known_concepts": len(self._concept_to_column),
        }
