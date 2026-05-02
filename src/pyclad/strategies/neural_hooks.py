from collections import OrderedDict
import inspect
import logging
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from pyclad.models.model import Model
from pyclad.models.neural_model import NeuralTrainableModel, resolve_torch_device

BatchLoss = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
DataTransform = Callable[[np.ndarray], np.ndarray]
logger = logging.getLogger(__name__)


class NeuralStrategyHooks:
    """Adapter for the common hooks exposed by torch-backed pyCLAD models."""

    _module_attribute_names = ("module", "model", "model_")

    def __init__(
        self,
        model: NeuralTrainableModel,
        *,
        module: Optional[torch.nn.Module] = None,
        data_transform: Optional[DataTransform] = None,
    ):
        if module is not None and not isinstance(module, torch.nn.Module):
            raise TypeError("Expected 'module' to be a torch.nn.Module.")
        self.model = model
        self._module = module
        self._explicit_module = module is not None
        self._data_transform = self.identity_transform if data_transform is None else data_transform

    def validate_trainable_model(self, strategy_name: str) -> None:
        """Fail early when a gradient-based strategy receives a classical model."""
        if self._explicit_module:
            return

        current = self.model
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if callable(getattr(current, "module_getter", None)):
                return

            for attr_name in self._module_attribute_names:
                if isinstance(getattr(current, attr_name, None), torch.nn.Module):
                    return

            current = getattr(current, "model", None)

        raise TypeError(
            f"{strategy_name} requires a neural model exposing a trainable torch module via "
            "module_getter(), '.module', '.model', or '.model_'. Classical PyOD models such as "
            "LOF, OCSVM, IForest, COPOD, and ECOD are not compatible with this gradient-based strategy."
        )

    def prepare_fit(
        self, data: np.ndarray, replay_data: Optional[np.ndarray] = None, model: Optional[Model] = None
    ) -> None:
        target = self.model if model is None else model
        prepare_fit = getattr(target, "prepare_fit", None)
        if callable(prepare_fit):
            try:
                prepare_fit(data, replay_data=replay_data)
            except TypeError:
                prepare_fit(data)
            return

        prepare_data = getattr(target, "prepare_data", None)
        if callable(prepare_data):
            try:
                prepare_data(data, fit=True)
            except TypeError:
                pass

    def prepare_data(self, data: np.ndarray, model: Optional[Model] = None) -> torch.Tensor:
        target = self.model if model is None else model
        prepare_data = getattr(target, "prepare_data", None)
        if callable(prepare_data):
            prepared = prepare_data(data)
            if torch.is_tensor(prepared):
                return prepared.float()
            return torch.as_tensor(prepared, dtype=torch.float32)

        transformed = self._transform_data_for_model(np.asarray(data), target)
        return torch.as_tensor(self._data_transform(transformed), dtype=torch.float32)

    def resolve_module(self, model: Optional[Model] = None) -> Optional[torch.nn.Module]:
        target = self.model if model is None else model
        if target is self.model and self._explicit_module and self._module is not None:
            return self._module

        current = target
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))

            module_getter = getattr(current, "module_getter", None)
            if callable(module_getter):
                candidate = module_getter()
                if isinstance(candidate, torch.nn.Module):
                    if target is self.model:
                        self._module = candidate
                    return candidate

            for attr_name in self._module_attribute_names:
                candidate = getattr(current, attr_name, None)
                if isinstance(candidate, torch.nn.Module):
                    if target is self.model:
                        self._module = candidate
                    return candidate

            current = getattr(current, "model", None)

        return None

    def resolve_device(
        self,
        model: Optional[Model] = None,
        explicit_device: Optional[Union[torch.device, str]] = None,
    ) -> torch.device:
        if explicit_device is not None:
            return resolve_torch_device(explicit_device)

        target = self.model if model is None else model
        model_device = self.resolve_attr("device", None, target)
        if model_device is not None:
            return resolve_torch_device(model_device)

        module = self.resolve_module(target)
        if module is not None:
            try:
                return next(module.parameters()).device
            except StopIteration:
                pass

        return resolve_torch_device()

    def resolve_attr(self, name: str, default, model: Optional[Model] = None):
        current = self.model if model is None else model
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if hasattr(current, name):
                return getattr(current, name)

            for attr_name in self._module_attribute_names:
                module = getattr(current, attr_name, None)
                if module is not None and hasattr(module, name):
                    return getattr(module, name)

            current = getattr(current, "model", None)

        return default

    def batch_loss(
        self,
        module: torch.nn.Module,
        batch: torch.Tensor,
        model: Optional[Model] = None,
        explicit_loss_fn: Optional[BatchLoss] = None,
    ) -> torch.Tensor:
        if explicit_loss_fn is not None:
            return self.ensure_scalar_loss(explicit_loss_fn(module, batch))

        target = self.model if model is None else model
        compute_loss = getattr(target, "compute_loss", None)
        if callable(compute_loss):
            return self.ensure_scalar_loss(compute_loss(module, batch))

        return self.ensure_scalar_loss(self.default_loss_fn(module, batch))

    def predict(self, data: np.ndarray, model: Optional[Model] = None) -> Tuple[np.ndarray, np.ndarray]:
        target = self.model if model is None else model
        predictions = target.predict(data)
        if isinstance(predictions, tuple) and len(predictions) == 2:
            return predictions

        decision_function = getattr(target, "decision_function", None)
        if callable(decision_function):
            return predictions, decision_function(data)

        return predictions, np.asarray(predictions)

    def batch_size(self, default: int = 32, model: Optional[Model] = None) -> int:
        batch_size = self.resolve_attr("batch_size", default, model)
        return int(batch_size) if batch_size else default

    def lr(self, default: float = 1e-2, model: Optional[Model] = None) -> float:
        lr = self.resolve_attr("lr", default, model)
        if lr == default:
            lr = self.resolve_attr("learning_rate", lr, model)
        return float(lr) if lr else default

    def epochs(self, default: int = 20, model: Optional[Model] = None) -> int:
        epochs = self.resolve_attr("epochs", default, model)
        if epochs == default:
            epochs = self.resolve_attr("epoch_num", epochs, model)
        return int(epochs) if epochs else default

    def shuffle(self, default: bool = True, model: Optional[Model] = None) -> bool:
        current = self.model if model is None else model
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))

            current_name = current.__class__.__name__
            if current_name in {"TemporalAutoencoder", "VariationalTemporalAutoencoder"}:
                return False
            if current_name == "Autoencoder":
                return True

            module = getattr(current, "module", None)
            if module is not None:
                module_name = module.__class__.__name__
                if module_name in {"TemporalAutoencoderModule", "VariationalTemporalAutoencoderModule"}:
                    return False
                if module_name == "AutoencoderModule":
                    return True

            current = getattr(current, "model", None)

        return default

    def drop_last(self, model: Optional[Model] = None) -> bool:
        target = self.model if model is None else model
        drop_last = getattr(target, "drop_last", None)
        if callable(drop_last):
            return bool(drop_last())
        return False

    def fit_config(self, model: Optional[Model] = None) -> Dict[str, Union[float, int, bool]]:
        return {
            "lr": self.lr(1e-2, model),
            "epochs": self.epochs(20, model),
            "batch_size": self.batch_size(32, model),
            "shuffle": self.shuffle(True, model),
            "drop_last": self.drop_last(model),
        }

    def model_name(self, model: Optional[Model] = None) -> str:
        target = self.model if model is None else model
        name = getattr(target, "name", None)
        return name() if callable(name) else target.__class__.__name__

    def threshold(self, model: Optional[Model] = None) -> Optional[float]:
        target = self.model if model is None else model
        for attr_name in ("threshold", "threshold_"):
            threshold = getattr(target, attr_name, None)
            if isinstance(threshold, (int, float, np.floating)):
                return float(threshold)
        return None

    def contamination(self, model: Optional[Model] = None) -> Optional[float]:
        target = self.model if model is None else model
        contamination = getattr(target, "contamination", None)
        if not isinstance(contamination, (int, float, np.floating)):
            return None
        contamination = float(contamination)
        if contamination <= 0.0 or contamination >= 1.0:
            return None
        return contamination

    def reconstruction_decoder(self, model: Optional[Model] = None) -> torch.nn.Module:
        target = self.model if model is None else model
        module = self.resolve_module(target)
        decoder = getattr(module, "decoder", None)
        if isinstance(decoder, torch.nn.Module):
            return decoder

        sequential = getattr(module, "model", None)
        if isinstance(sequential, torch.nn.Sequential) and "net_output" in sequential._modules:
            if not bool(getattr(target, "use_ae", False)):
                raise ValueError("PNNStrategy supports ContinualDeepSVDD only with use_ae=True.")

            tail = OrderedDict()
            use_tail = False
            for name, layer in sequential._modules.items():
                if use_tail:
                    tail[name] = layer
                elif name == "net_output":
                    use_tail = True

            if tail:
                return torch.nn.Sequential(tail)

        raise TypeError(
            "Model must expose a reconstruction decoder. Supported lazy neural models are "
            "ContinualVAE, ContinualAE1SVM, and ContinualDeepSVDD(use_ae=True)."
        )

    def after_fit(self, data: np.ndarray, model: Optional[Model] = None) -> None:
        target = self.model if model is None else model
        if hasattr(target, "_auto_threshold") and target._auto_threshold:
            target._calibrate_threshold(data)
            return

        after_fit = getattr(target, "after_fit", None)
        if callable(after_fit):
            after_fit(data)

    def forward_batch(self, batch: torch.Tensor, model: Optional[Model] = None):
        target = self.model if model is None else model
        forward_batch = getattr(target, "forward_batch", None)
        if callable(forward_batch):
            return forward_batch(batch, apply_masking=False)

        module = self.resolve_module(target)
        if module is None:
            raise TypeError("Model does not expose a trainable torch module or forward_batch().")

        output = module(batch)
        x_hat = output[0] if isinstance(output, (tuple, list)) else output
        z = None
        try:
            z = self.encode_batch(batch, target)
        except TypeError:
            pass
        return x_hat, z, None

    def encode_batch(self, batch: torch.Tensor, model: Optional[Model] = None) -> torch.Tensor:
        target = self.model if model is None else model
        encode_batch = getattr(target, "encode_batch", None)
        if callable(encode_batch):
            return self.extract_tensor(encode_batch(batch))

        module = self.resolve_module(target)
        if module is None:
            raise TypeError("Model does not expose a trainable torch module or encode_batch().")

        if hasattr(module, "encode"):
            return self.extract_tensor(module.encode(batch))

        if hasattr(module, "encoder") and hasattr(module, "encoder_mu"):
            return self.extract_tensor(module.encoder_mu(module.encoder(batch)))

        if hasattr(module, "encoder"):
            return self.extract_tensor(module.encoder(batch))

        raise TypeError("Model does not expose encoder hooks required for latent distillation.")

    def _transform_data_for_model(self, data: np.ndarray, model: Model) -> np.ndarray:
        current = model
        transformed = np.asarray(data)
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if current.__class__.__name__ == "FlattenTimeSeriesAdapter":
                transformed = transformed.reshape(transformed.shape[0], -1)
            current = getattr(current, "model", None)
        return transformed

    @staticmethod
    def identity_transform(data: np.ndarray) -> np.ndarray:
        return data

    @staticmethod
    def ensure_scalar_loss(loss: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss, dtype=torch.float32)
        if loss.ndim > 0:
            return loss.mean()
        return loss

    @staticmethod
    def default_loss_fn(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        output = module(batch)
        x_hat = output[0] if isinstance(output, (tuple, list)) else output
        extras = tuple(output[1:]) if isinstance(output, (tuple, list)) else tuple()
        train_loss = getattr(module, "train_loss", None)
        if callable(train_loss):
            loss = NeuralStrategyHooks._call_train_loss_if_unambiguous(train_loss, batch, x_hat, extras)
            if loss is not None:
                return loss

        if x_hat.shape != batch.shape:
            raise ValueError(
                "Default neural strategy loss expects the module output to have the same shape as the input batch. "
                "Expose compute_loss() or pass a custom loss_fn for models with a different training objective."
            )
        logger.debug("Falling back to plain MSE for %s.", module.__class__.__name__)
        return torch.nn.functional.mse_loss(x_hat, batch)

    @staticmethod
    def _call_train_loss_if_unambiguous(train_loss, batch: torch.Tensor, x_hat: torch.Tensor, extras: tuple):
        if extras:
            for args in ((batch, x_hat, *extras), (x_hat, batch, *extras)):
                if NeuralStrategyHooks._can_bind_loss_args(train_loss, args):
                    return train_loss(*args)
            return None

        if isinstance(train_loss, torch.nn.Module):
            return train_loss(x_hat, batch)

        names = NeuralStrategyHooks._loss_parameter_names(train_loss)
        if len(names) >= 2 and names[0] in {"x_hat", "prediction", "pred", "output"} and names[1] in {
            "x",
            "target",
            "batch",
            "input",
        }:
            return train_loss(x_hat, batch)
        if len(names) >= 2 and names[0] in {"x", "target", "batch", "input"} and names[1] in {
            "x_hat",
            "prediction",
            "pred",
            "output",
        }:
            return train_loss(batch, x_hat)
        return None

    @staticmethod
    def _can_bind_loss_args(train_loss, args: tuple) -> bool:
        try:
            inspect.signature(train_loss).bind(*args)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _loss_parameter_names(train_loss) -> list:
        try:
            signature = inspect.signature(train_loss)
        except (TypeError, ValueError):
            return []
        return [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]

    @staticmethod
    def extract_tensor(output):
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if torch.is_tensor(output):
            return output
        raise TypeError("Expected a tensor or a sequence containing a tensor.")
