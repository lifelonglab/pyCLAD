import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.torch_backbone import TorchBackbone


class TorchModelAdapter(Model):
    """Adapts a TorchBackbone to the Model interface.

    Allows backbone-based models to be used with model-agnostic strategies
    (e.g. ReplayOnlyStrategy, ReplayEnhancedStrategy) that drive training
    through Model.fit().
    """

    def __init__(self, backbone: TorchBackbone, epochs: int, batch_size: int):
        self._backbone = backbone
        self._epochs = epochs
        self._batch_size = batch_size

    def fit(self, data: np.ndarray) -> None:
        loader = DataLoader(
            TensorDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=self._batch_size,
            shuffle=True,
        )
        self._backbone.fit_with_loss(
            loader,
            lambda batch: self._backbone.compute_loss(batch[0]),
            self._epochs,
        )

    def predict(self, data: np.ndarray):  # return type follows backbone
        return self._backbone.predict(data)

    def name(self) -> str:
        return self._backbone.name()

    def additional_info(self) -> dict:
        return {**self._backbone.additional_info(), "epochs": self._epochs, "batch_size": self._batch_size}
