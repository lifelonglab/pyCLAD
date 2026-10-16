# requires: torch
# --8<-- [start:backbone]
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from pyclad.models.torch_backbone import TorchBackbone
from pyclad.output.prediction_results import PredictionResults


class LinearAutoencoder(TorchBackbone):
    """A one-layer autoencoder; anomaly score = reconstruction error."""

    def __init__(self, n_features: int, n_hidden: int, lr: float = 1e-2, threshold: float = 0.5):
        self.module = nn.Sequential(nn.Linear(n_features, n_hidden), nn.Linear(n_hidden, n_features))
        self.n_features, self.n_hidden, self.lr, self.threshold = n_features, n_hidden, lr, threshold

    def get_module(self) -> nn.Module:
        return self.module

    def get_optimizer(self) -> Optimizer:
        # A fresh optimizer each time: strategies call this once per training run.
        return torch.optim.Adam(self.module.parameters(), lr=self.lr)

    def compute_loss(self, x: Tensor) -> Tensor:
        return F.mse_loss(self.module(x), x)  # keeps grad_fn: strategies call backward() on it

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def predict(self, data: np.ndarray) -> PredictionResults:
        # Not declared by TorchBackbone, but TorchModelAdapter and the strategies call it.
        self.module.eval()
        with torch.no_grad():
            reconstruction = self.forward(torch.tensor(data, dtype=torch.float32)).numpy()
        self.module.train()
        errors = ((data - reconstruction) ** 2).mean(axis=1)
        return PredictionResults(y_pred=(errors > self.threshold).astype(int), anomaly_scores=errors)

    def name(self) -> str:
        return "LinearAutoencoder"

    def additional_info(self):
        return {"n_features": self.n_features, "n_hidden": self.n_hidden, "lr": self.lr, "threshold": self.threshold}


# --8<-- [end:backbone]

# --8<-- [start:use]
from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.regularization.ewc import EWCStrategy

backbone = LinearAutoencoder(n_features=6, n_hidden=2)
model = TorchModelAdapter(backbone, StandardRunner(max_epochs=5), batch_size=32)  # for any strategy
ewc = EWCStrategy(model=LinearAutoencoder(n_features=6, n_hidden=2), runner=StandardRunner(max_epochs=5))
# --8<-- [end:use]

data = np.random.default_rng(0).normal(0.5, 0.1, size=(200, 6))
model.fit(data)
ewc.learn(data)
print(model.predict(data[:3]).anomaly_scores, ewc.predict(data[:3]).anomaly_scores)
