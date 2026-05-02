from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import torch
from sklearn.utils.validation import check_array
from torch import nn

from pyclad.models.model import Model


def resolve_torch_device(device: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device not in (None, "auto"):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lightning_trainer_device_kwargs(device: Optional[Union[str, torch.device]] = "auto") -> dict:
    resolved = resolve_torch_device(device)
    if resolved.type == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    if resolved.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu"}


class PyCLADModelProtocol(Protocol):
    def fit(self, data: np.ndarray): ...

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

    def name(self) -> str: ...


@runtime_checkable
class ModuleGetterNeuralModel(PyCLADModelProtocol, Protocol):
    def module_getter(self) -> Optional[nn.Module]: ...


@runtime_checkable
class ModuleAttributeNeuralModel(PyCLADModelProtocol, Protocol):
    module: nn.Module


@runtime_checkable
class ModelAttributeNeuralModel(PyCLADModelProtocol, Protocol):
    model: nn.Module


@runtime_checkable
class FittedModelAttributeNeuralModel(PyCLADModelProtocol, Protocol):
    model_: nn.Module


class NeuralModel(Model):
    """Shared lifecycle hooks for PyOD-style torch-backed continual neural models."""

    _module_attribute_names = ("module", "model", "model_")

    def fit(self, X, y=None):
        X = check_array(X)
        self.prepare_fit(X, y=y)
        train_set = self.fit_dataset(X, y=y)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.fit_drop_last(),
        )
        self.train_fit_loader(train_loader)
        return self.after_fit(X)

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.decision_function(data)
        threshold = getattr(self, "threshold_", getattr(self, "threshold", None))
        if threshold is None:
            raise ValueError(f"{self.__class__.__name__} has no fitted anomaly threshold.")
        return (scores > threshold).astype(int), scores

    def prepare_fit(self, X, replay_data=None, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        self.set_input_shape(X)
        fit_data = self.combined_fit_data(X, replay_data)
        self.prepare_preprocessing(fit_data)
        self.ensure_module(X.shape[1])
        self.after_prepare_fit(X, fit_data)

    def prepare_data(self, X, fit=False):
        X = check_array(X)
        if fit:
            self.prepare_fit(X)
        if self.module_getter() is None:
            raise ValueError(
                f"Neural data preparation requires the {self.__class__.__name__} model to be built first."
            )
        X = self.transform_prepared_data(X)
        return torch.as_tensor(X, dtype=torch.float32)

    def module_getter(self):
        for attr_name in self._module_attribute_names:
            module = getattr(self, attr_name, None)
            if module is not None:
                return module
        return None

    @staticmethod
    def combined_fit_data(current_data, replay_data):
        current_data = check_array(current_data)
        if replay_data is None or len(replay_data) == 0:
            return current_data
        return np.concatenate([check_array(replay_data), current_data], axis=0)

    @staticmethod
    def safe_std(std):
        return np.where(std < 1e-6, 1.0, std)

    @staticmethod
    def clone_state_dict(state_dict):
        return {key: value.detach().clone() for key, value in state_dict.items()}

    def after_fit(self, X):
        state_dict = self.snapshot_model_state()
        if state_dict is not None:
            self.best_model_dict = state_dict
        X = check_array(X)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def _set_n_classes(self, y=None):
        parent_hook = getattr(super(NeuralModel, self), "_set_n_classes", None)
        if callable(parent_hook):
            parent_hook(y)

    def _process_decision_scores(self):
        parent_hook = getattr(super(NeuralModel, self), "_process_decision_scores", None)
        if callable(parent_hook):
            parent_hook()

    def drop_last(self):
        return False

    def name(self) -> str:
        return self.__class__.__name__

    def info(self) -> Dict[str, Any]:
        return {"model": {"name": self.name(), **self.additional_info()}}

    def additional_info(self):
        return {}

    @abstractmethod
    def set_input_shape(self, X):
        raise NotImplementedError

    @abstractmethod
    def prepare_preprocessing(self, fit_data):
        raise NotImplementedError

    @abstractmethod
    def ensure_module(self, n_features):
        raise NotImplementedError

    @abstractmethod
    def fit_dataset(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def train_fit_loader(self, train_loader):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, module, batch):
        raise NotImplementedError

    def forward_batch(self, batch, apply_masking=False):
        del apply_masking
        module = self.module_getter()
        if module is None:
            raise TypeError(f"{self.__class__.__name__} does not expose a trainable torch module.")

        output = module(batch)
        x_hat = output[0] if isinstance(output, (tuple, list)) else output
        z = None
        try:
            z = self.encode_batch(batch)
        except TypeError:
            pass
        return x_hat, z, None

    def encode_batch(self, batch):
        module = self.module_getter()
        if module is None:
            raise TypeError(f"{self.__class__.__name__} does not expose a trainable torch module.")

        if hasattr(self, "forward_with_representation"):
            _, representation = self.forward_with_representation(module, batch)
            return representation

        if hasattr(module, "encode"):
            return self._extract_tensor(module.encode(batch))

        if hasattr(module, "encoder") and hasattr(module, "encoder_mu"):
            return self._extract_tensor(module.encoder_mu(module.encoder(batch)))

        if hasattr(module, "encoder"):
            return self._extract_tensor(module.encoder(batch))

        raise TypeError(f"{self.__class__.__name__} does not expose an encoder hook.")

    def after_prepare_fit(self, X, fit_data):
        pass

    def snapshot_model_state(self) -> Optional[Dict[str, torch.Tensor]]:
        module = self.module_getter()
        if module is None:
            return None
        return self.clone_state_dict(module.state_dict())

    def fit_drop_last(self) -> bool:
        return False

    def transform_prepared_data(self, X):
        if not getattr(self, "preprocessing", False):
            return np.copy(X)

        if hasattr(self, "scaler_"):
            return self.scaler_.transform(X)

        if hasattr(self, "X_mean") and hasattr(self, "X_std"):
            return (X - self.X_mean) / (self.X_std + 1e-8)

        if hasattr(self, "mean") and hasattr(self, "std"):
            return (X - self.mean) / (self.std + 1e-8)

        raise ValueError(f"{self.__class__.__name__} preprocessing has not been prepared.")

    @staticmethod
    def _extract_tensor(output):
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if torch.is_tensor(output):
            return output
        raise TypeError("Expected a tensor or a sequence containing a tensor.")


NeuralTrainableModel = Union[
    NeuralModel,
    ModuleGetterNeuralModel,
    ModuleAttributeNeuralModel,
    ModelAttributeNeuralModel,
    FittedModelAttributeNeuralModel,
]


__all__ = [
    "NeuralModel",
    "NeuralTrainableModel",
    "resolve_torch_device",
    "lightning_trainer_device_kwargs",
    "ModuleGetterNeuralModel",
    "ModuleAttributeNeuralModel",
    "ModelAttributeNeuralModel",
    "FittedModelAttributeNeuralModel",
]
