"""Progressive Neural Networks architectural strategy for anomaly detection."""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import TensorDataset

from pyclad.models.neural_model import NeuralTrainableModel, lightning_trainer_device_kwargs, resolve_torch_device
from pyclad.strategies.neural_hooks import NeuralStrategyHooks
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy

AdapterFactory = Callable[[torch.Size, torch.Size], nn.Module]


class _HookEncoder(nn.Module):
    """PNN encoder adapter backed by a model's shared encode_batch hook."""

    def __init__(self, model: NeuralTrainableModel, hooks: NeuralStrategyHooks, module: nn.Module):
        super().__init__()
        self.model = model
        self.hooks = hooks
        self.source_module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hooks.encode_batch(x, self.model)


class _PNNAutoencoderColumn(nn.Module):
    """A single frozen-or-trainable PNN autoencoder column."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_previous_columns: int,
        adapter_factory: Optional[AdapterFactory],
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_previous_columns = num_previous_columns
        self.adapter_factory = adapter_factory
        self.adapters = nn.ModuleList()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _make_adapter(self, old_shape: torch.Size, new_shape: torch.Size, device: torch.device) -> nn.Module:
        if self.adapter_factory is None:
            if old_shape != new_shape:
                raise ValueError(
                    "Identity adapters require matching latent shapes. "
                    f"Received old latent shape {tuple(old_shape)} and new latent shape {tuple(new_shape)}. "
                    "Provide adapter_factory to define a projection."
                )
            return nn.Identity().to(device)

        adapter = self.adapter_factory(old_shape, new_shape)
        if not isinstance(adapter, nn.Module):
            raise TypeError("adapter_factory must return an nn.Module instance.")
        return adapter.to(device)

    def _ensure_adapters(self, old_latents: List[torch.Tensor], z_new: torch.Tensor) -> None:
        if len(old_latents) != self.num_previous_columns:
            raise ValueError(f"Column expects {self.num_previous_columns} old latents but received {len(old_latents)}.")
        if self.adapters:
            return

        self.adapters.extend(
            self._make_adapter(old_latent.shape[1:], z_new.shape[1:], z_new.device) for old_latent in old_latents
        )

    def forward(
        self, x: torch.Tensor, old_latents: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_latents = [] if old_latents is None else list(old_latents)
        if len(old_latents) != self.num_previous_columns:
            raise ValueError(f"Column expects {self.num_previous_columns} old latents but received {len(old_latents)}.")

        z_new = self.encoder(x)
        if not torch.is_tensor(z_new):
            raise TypeError("PNNStrategy expects encoders to return a tensor latent representation.")

        if old_latents:
            self._ensure_adapters(old_latents, z_new)
            lateral = torch.stack(
                [adapter(old_latent) for adapter, old_latent in zip(self.adapters, old_latents)], dim=0
            ).sum(dim=0)
            z_new = z_new + lateral

        return self.decoder(z_new), z_new


class _PNNAutoencoderModule(pl.LightningModule):
    """Internal Lightning module that trains the active PNN autoencoder column."""

    def __init__(self, adapter_factory: Optional[AdapterFactory], freeze_old_columns: bool, lr: float = 1e-3):
        super().__init__()
        self.adapter_factory = adapter_factory
        self.freeze_old_columns = freeze_old_columns
        self.lr = lr
        self.columns = nn.ModuleList()
        self.current_task = -1
        self.loss_fn = nn.MSELoss()

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    def _current_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def add_column(self, encoder: nn.Module, decoder: nn.Module) -> None:
        if self.freeze_old_columns:
            for column in self.columns:
                column.freeze()

        self.columns.append(
            _PNNAutoencoderColumn(
                encoder=encoder,
                decoder=decoder,
                num_previous_columns=self.num_columns,
                adapter_factory=self.adapter_factory,
            ).to(self._current_device())
        )
        self.current_task = self.num_columns - 1

    def _old_latents(self, x: torch.Tensor, task_label: int) -> List[torch.Tensor]:
        latents = []
        for column in self.columns[:task_label]:
            with torch.no_grad():
                _, latent = column(x, latents)
            latents.append(latent.detach())
        return latents

    def forward(self, x: torch.Tensor, task_label: Optional[int] = None) -> torch.Tensor:
        task_label = self.current_task if task_label is None else task_label
        if task_label < 0 or task_label >= self.num_columns:
            raise ValueError(f"Invalid task label {task_label}. Available tasks: 0..{self.num_columns - 1}.")

        x_hat, _ = self.columns[task_label](x, self._old_latents(x, task_label))
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.loss_fn(self(x), x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, self.parameters()), lr=self.lr)


class PNNStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Progressive Neural Networks using reconstruction error for anomaly scores."""

    def __init__(
        self,
        base_model_factory: Callable[[], NeuralTrainableModel],
        adapter_factory: Optional[AdapterFactory] = None,
        task_free: bool = False,
        freeze_old_columns: bool = True,
        threshold: Optional[float] = None,
        auto_add_column: bool = False,
        random_state: Optional[int] = None,
        device: Union[str, torch.device, None] = "auto",
    ):
        if not callable(base_model_factory):
            raise TypeError("base_model_factory must be callable and return a fresh autoencoder-style model.")

        self.base_model_factory = base_model_factory
        self.task_free = task_free
        self.freeze_old_columns = freeze_old_columns
        self.threshold = threshold
        self.auto_add_column = auto_add_column
        self.random_state = random_state
        self.device = resolve_torch_device(device)
        self.module = _PNNAutoencoderModule(adapter_factory, freeze_old_columns).to(self.device)
        self._configs: List[Dict[str, Union[float, int, bool]]] = []
        self._column_models: List[NeuralTrainableModel] = []
        self._column_hooks: List[NeuralStrategyHooks] = []
        self._trained_columns: Set[int] = set()
        self._concept_to_task: Dict[str, int] = {}
        self._base_model_name: Optional[str] = None
        self._threshold_is_fixed = threshold is not None
        self._pending_new_column = True

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self._try_add_eager_column()

    @property
    def current_task(self) -> int:
        return self.module.current_task

    @property
    def num_columns(self) -> int:
        return self.module.num_columns

    @staticmethod
    def _column_parts(
        model: NeuralTrainableModel, hooks: NeuralStrategyHooks, module: nn.Module
    ) -> Tuple[nn.Module, nn.Module, Dict[str, Union[float, int, bool]], Optional[float], str]:
        return (
            _HookEncoder(model, hooks, module),
            hooks.reconstruction_decoder(model),
            hooks.fit_config(model),
            hooks.threshold(model),
            hooks.model_name(model),
        )

    def _trainer(self, epochs: int) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=epochs,
            **lightning_trainer_device_kwargs(self.device),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    def _current_device(self) -> torch.device:
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _try_add_eager_column(self) -> None:
        try:
            model, hooks, module = self._new_model_context()
            if module is None:
                return
            parts = self._column_parts(model, hooks, module)
        except (TypeError, ValueError):
            return

        self._add_column(model, hooks, *parts)
        self._pending_new_column = False

    def _new_model_context(
        self, data: Optional[np.ndarray] = None
    ) -> Tuple[NeuralTrainableModel, NeuralStrategyHooks, Optional[nn.Module]]:
        model = self.base_model_factory()
        self._set_model_device(model)
        hooks = NeuralStrategyHooks(model)
        hooks.validate_trainable_model("PNNStrategy")
        if data is not None:
            hooks.prepare_fit(data, model=model)
        return model, hooks, hooks.resolve_module(model)

    def _set_model_device(self, model: NeuralTrainableModel) -> None:
        if hasattr(model, "device"):
            setattr(model, "device", self.device)

    def _prepared_tensor_for_task(self, data: np.ndarray, task_label: int) -> torch.Tensor:
        if task_label < 0 or task_label >= len(self._column_hooks):
            raise ValueError(f"Invalid task label {task_label}. Available tasks: 0..{len(self._column_hooks) - 1}.")
        hooks = self._column_hooks[task_label]
        model = self._column_models[task_label]
        prepared = hooks.prepare_data(data, model=model)
        return prepared.detach().float()

    def _build_column_from_data(self, data: np.ndarray) -> int:
        model, hooks, module = self._new_model_context(data)
        if module is None:
            raise TypeError("base_model_factory returned a model that does not expose a trainable torch module.")

        module.to(self.device)
        self._add_column(model, hooks, *self._column_parts(model, hooks, module))
        self._pending_new_column = False
        return self.current_task

    def _add_column(
        self,
        model: NeuralTrainableModel,
        hooks: NeuralStrategyHooks,
        encoder: nn.Module,
        decoder: nn.Module,
        config: Dict[str, Union[float, int, bool]],
        threshold: Optional[float],
        base_model_name: str,
    ) -> None:
        self._base_model_name = base_model_name
        if not self._threshold_is_fixed and threshold is not None:
            self.threshold = threshold

        self._column_models.append(model)
        self._column_hooks.append(hooks)
        self._configs.append(config)
        self.module.add_column(encoder, decoder)
        self.module.to(self.device)

    def _ensure_current_column(self, data: np.ndarray) -> int:
        if self.current_task >= 0 and not self._pending_new_column:
            return self.current_task
        return self._build_column_from_data(data)

    def _update_threshold_after_fit(self, data: np.ndarray, task_label: int) -> None:
        if self._threshold_is_fixed:
            return

        model = self._column_models[task_label]
        hooks = self._column_hooks[task_label]
        model_threshold = hooks.threshold(model)
        if model_threshold is not None:
            self.threshold = model_threshold
            return

        contamination = hooks.contamination(model)
        if contamination is not None:
            scores = self._scores_for_task(data, task_label)
            self.threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
            return

        if self.threshold is None:
            self.threshold = 0.5

    @staticmethod
    def _to_numpy(data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _reconstruction_error(data: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        return ((data - x_hat) ** 2).reshape(len(data), -1).mean(axis=1)

    def _scores_for_task(self, data: np.ndarray, task_label: int) -> np.ndarray:
        data = self._to_numpy(data)
        tensor_data = self._prepared_tensor_for_task(data, task_label)
        prepared_data = tensor_data.detach().cpu().numpy()
        tensor_data = tensor_data.to(self._current_device())
        self.module.eval()
        with torch.no_grad():
            x_hat = self.module(tensor_data, task_label=task_label).detach().cpu().numpy()
        return self._reconstruction_error(prepared_data, x_hat)

    def fit(self, data: np.ndarray) -> None:
        data = self._to_numpy(data)
        if len(data) == 0:
            raise ValueError("PNNStrategy.fit received an empty dataset.")

        task_label = self._ensure_current_column(data)
        config = self._configs[task_label]
        tensor_data = self._prepared_tensor_for_task(data, task_label).cpu()
        dataloader = torch.utils.data.DataLoader(
            TensorDataset(tensor_data),
            batch_size=int(config["batch_size"]),
            shuffle=bool(config["shuffle"]),
            drop_last=bool(config["drop_last"]),
        )

        self.module.lr = float(config["lr"])
        self.module.train()
        sample = tensor_data[: min(len(tensor_data), int(config["batch_size"]))].to(self._current_device())
        _ = self.module(sample, task_label=task_label)

        self._trainer(int(config["epochs"])).fit(self.module, dataloader)
        self.module.to(self.device)
        self._trained_columns.add(task_label)
        self._update_threshold_after_fit(data, task_label)

        if self.auto_add_column:
            self.end_task()

    def learn(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> None:
        data = self._to_numpy(data)
        if concept_id is None:
            if self.current_task in self._trained_columns and not self._pending_new_column:
                self.end_task()
        elif concept_id in self._concept_to_task:
            if self._concept_to_task[concept_id] != self.current_task or self._pending_new_column:
                raise ValueError(
                    f"Concept '{concept_id}' is already assigned to frozen task {self._concept_to_task[concept_id]}."
                )
        else:
            if self.current_task in self._trained_columns and not self._pending_new_column:
                self.end_task()
            self._concept_to_task[concept_id] = self._ensure_current_column(data)

        self.fit(data)

    def predict(
        self,
        data: np.ndarray,
        task_label: Optional[int] = None,
        concept_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.task_free and task_label is None and concept_id in self._concept_to_task:
            task_label = self._concept_to_task[concept_id]

        if task_label is None:
            task_labels = (
                sorted(self._trained_columns) if self.task_free and self._trained_columns else [self.current_task]
            )
            scores = [self._scores_for_task(data, label) for label in task_labels]
            final_scores = scores[0] if len(scores) == 1 else np.stack(scores, axis=0).min(axis=0)
        else:
            final_scores = self._scores_for_task(data, task_label)

        threshold = 0.5 if self.threshold is None else self.threshold
        return (final_scores > threshold).astype(int), final_scores

    def end_task(self) -> None:
        self._pending_new_column = True

    def name(self) -> str:
        return "PNN"

    def additional_info(self) -> Dict:
        return {
            "base_model": self._base_model_name,
            "task_free": self.task_free,
            "freeze_old_columns": self.freeze_old_columns,
            "threshold": self.threshold,
            "auto_add_column": self.auto_add_column,
            "random_state": self.random_state,
            "device": str(self.device),
            "current_task": self.current_task,
            "num_columns": self.num_columns,
            "trained_columns": len(self._trained_columns),
            "known_concepts": len(self._concept_to_task),
        }
