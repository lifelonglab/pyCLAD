import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import TensorDataset

from pyclad.models.autoencoder.loss import VariationalMSELoss
from pyclad.models.model import Model
from pyclad.models.torch_backbone import TorchBackbone
from pyclad.output.prediction_results import PredictionResults


class Autoencoder(TorchBackbone):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-2,
        threshold: float = 0.5,
    ):
        self.module = AutoencoderModule(encoder, decoder)
        self.lr = lr
        self.threshold = threshold

    def get_module(self) -> nn.Module:
        return self.module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.Adam(self.module.parameters(), lr=self.lr)

    def compute_loss(self, x: Tensor) -> Tensor:
        return F.mse_loss(self.module(x), x)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def predict(self, data: np.ndarray) -> PredictionResults:
        self.module.eval()
        with torch.no_grad():
            x_hat = self.forward(torch.tensor(data, dtype=torch.float32)).numpy()
        rec_error = ((data - x_hat) ** 2).mean(axis=1)
        return PredictionResults(
            y_pred=(rec_error > self.threshold).astype(int),
            anomaly_scores=rec_error,
        )

    def name(self) -> str:
        return "Autoencoder"

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "lr": self.lr,
        }


class AutoencoderModule(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariationalAutoencoder(TorchBackbone):
    """Variational autoencoder backbone (VAE analog of :class:`Autoencoder`).

    The ``encoder`` maps an input to a hidden representation of size ``hidden_dim``;
    two linear heads then project it to the ``latent_dim`` mean and log-variance of the
    approximate posterior. The ``decoder`` reconstructs the input from a latent sample.

    :param hidden_dim: feature size produced by ``encoder`` (input to the mean/log-var heads).
    :param latent_dim: dimensionality of the latent space (input to ``decoder``).
    :param kl_weight: weight on the KL term (``beta`` in beta-VAE); 1.0 is the standard VAE.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_dim: int,
        latent_dim: int,
        lr: float = 1e-2,
        threshold: float = 0.5,
        kl_weight: float = 1.0,
    ):
        self.module = VariationalAutoencoderModule(encoder, decoder, hidden_dim, latent_dim)
        self.lr = lr
        self.threshold = threshold
        self.kl_weight = kl_weight

    def get_module(self) -> nn.Module:
        return self.module

    def get_optimizer(self) -> Optimizer:
        return torch.optim.Adam(self.module.parameters(), lr=self.lr)

    def compute_loss(self, x: Tensor) -> Tensor:
        x_hat, mean, log_var = self.module(x)
        reconstruction = F.mse_loss(x_hat, x)
        # KL[N(mean, exp(log_var)) || N(0, 1)], averaged over batch and latent dims.
        kl_divergence = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstruction + self.kl_weight * kl_divergence

    def forward(self, x: Tensor) -> Tensor:
        x_hat, _, _ = self.module(x)
        return x_hat

    def predict(self, data: np.ndarray) -> PredictionResults:
        self.module.eval()
        with torch.no_grad():
            # Score with the deterministic latent mean so anomaly scores are not noisy across calls.
            x_hat = self.module.reconstruct(torch.tensor(data, dtype=torch.float32)).numpy()
        rec_error = ((data - x_hat) ** 2).mean(axis=1)
        return PredictionResults(
            y_pred=(rec_error > self.threshold).astype(int),
            anomaly_scores=rec_error,
        )

    def name(self) -> str:
        return "VariationalAutoencoder"

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "hidden_dim": self.module.mean_layer.in_features,
            "latent_dim": self.module.mean_layer.out_features,
            "lr": self.lr,
            "kl_weight": self.kl_weight,
        }


class VariationalAutoencoderModule(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mean_layer(h), self.logvar_layer(h)

    @staticmethod
    def reparametrize(mean: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_var = self.encode(x)
        z = self.reparametrize(mean, log_var)
        return self.decoder(z), mean, log_var

    def reconstruct(self, x: Tensor) -> Tensor:
        """Deterministic reconstruction from the latent mean (no sampling)."""
        mean, _ = self.encode(x)
        return self.decoder(mean)


class TemporalAutoencoder(Model):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-2,
        threshold: float = 0.5,
        epochs: int = 20,
    ):
        self.module = TemporalAutoencoderModule(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs

    def fit(self, data: np.ndarray):
        dataset = TensorDataset(torch.Tensor(data))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self.module, dataloader)

    def predict(self, data: np.ndarray) -> PredictionResults:
        batch_size, seq_len, input_size = data.shape
        x_hat = self.module(torch.Tensor(data)).detach()
        rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=2)
        rec_error = rec_error.reshape((batch_size, seq_len, 1))
        return PredictionResults(
            y_pred=(rec_error > self.threshold).astype(int),
            anomaly_scores=rec_error,
        )

    def name(self) -> str:
        return "TemporalAutoencoder"

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "lr": self.module.lr,
            "epochs": self.epochs,
        }


class TemporalAutoencoderModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

        self.save_hyperparameters()
        self.train_loss = nn.MSELoss()
        self.val_loss = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.train_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.val_loss(x_hat, x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class VariationalTemporalAutoencoder(Model):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-2,
        threshold: float = 0.5,
        epochs: int = 20,
    ):
        self.module = VariationalTemporalAutoencoderModule(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs

    def fit(self, data: np.ndarray):
        dataset = TensorDataset(torch.Tensor(data))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self.module, dataloader)

    def predict(self, data: np.ndarray) -> PredictionResults:
        batch_size, seq_len, input_size = data.shape
        x_hat, mean, var = self.module(torch.Tensor(data))
        x_hat = x_hat.detach()
        rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=2)
        rec_error = rec_error.reshape((batch_size, seq_len, 1))
        return PredictionResults(
            y_pred=(rec_error > self.threshold).astype(int),
            anomaly_scores=rec_error,
        )

    @staticmethod
    def create_sequences(data: np.ndarray, seq_len: int, step: int = 1) -> np.ndarray:
        sequences = []
        for i in range(0, len(data) - seq_len + 1, step):
            sequences.append(data[i : i + seq_len])
        return np.stack(sequences)

    def name(self) -> str:
        return "VariationalTemporalAutoencoder"

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "lr": self.module.lr,
            "epochs": self.epochs,
        }


class VariationalTemporalAutoencoderModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

        self.save_hyperparameters()
        self.train_loss = VariationalMSELoss()
        self.val_loss = VariationalMSELoss()

    def forward(self, x):
        mean, var = self.encoder(x)
        x = self.reparametrize(mean, var)
        x = self.decoder(x)
        return x, mean, var

    @staticmethod
    def reparametrize(mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, mean, var = self(x)
        loss = self.train_loss(x_hat, x, mean, var)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, mean, var = self(x)
        loss = self.val_loss(x_hat, x, mean, var)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
