import numpy as np

from pyclad.models.model import Model
from pyclad.models.torch_backbone import TorchBackbone
from pyclad.models.training.loaders import float_tensor_loader
from pyclad.models.training.runners.runner import TorchRunner


class TorchModelAdapter(Model):
    """Adapts a TorchBackbone to the Model interface.

    Allows backbone-based models to be used with model-agnostic strategies
    (e.g. ReplayOnlyStrategy, ReplayEnhancedStrategy) that drive training
    through Model.fit(). The supplied runner owns the training loop, so the
    same adapter works with plain or early-stopping training.
    """

    def __init__(self, backbone: TorchBackbone, runner: TorchRunner, batch_size: int):
        self._backbone = backbone
        self._runner = runner
        self._batch_size = batch_size

    def fit(self, data: np.ndarray) -> None:
        train, val = self._runner.split_train_test(data)
        loss_fn = lambda batch: self._backbone.compute_loss(batch[0])  # noqa: E731
        self._runner.run(
            self._backbone,
            float_tensor_loader(train, self._batch_size, shuffle=True),
            loss_fn,
            val_loader=float_tensor_loader(val, self._batch_size, shuffle=False) if val is not None else None,
            val_loss_fn=loss_fn,
        )

    def predict(self, data: np.ndarray):  # return type follows backbone
        return self._backbone.predict(data)

    def name(self) -> str:
        return self._backbone.name()

    def additional_info(self) -> dict:
        return {**self._backbone.additional_info(), "batch_size": self._batch_size, "runner": self._runner.info()}
