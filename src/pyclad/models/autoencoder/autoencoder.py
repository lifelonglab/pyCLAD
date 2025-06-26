import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import TensorDataset

from pyclad.models.autoencoder.loss import VariationalMSELoss
from pyclad.models.model import Model


class Autoencoder(Model):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-2, threshold: float = 0.5, epochs: int = 20
    ):
        self.module = AutoencoderModule(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs

    def fit(self, data: np.ndarray):
        dataset = TensorDataset(torch.Tensor(data))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self.module, dataloader)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        x_hat = self.module(torch.Tensor(data)).detach()
        rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=1)

        binary_predictions = (rec_error > self.threshold).astype(int)
        return binary_predictions, rec_error

    def name(self) -> str:
        return "Autoencoder"

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "lr": self.module.lr,
            "epochs": self.epochs,
        }


class AutoencoderModule(pl.LightningModule):
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


class TemporalAutoencoder(Model):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-2,
        threshold: float = 0.5,
        epochs: int = 20,
        seq_len: int = 10,
        seq_step: int = 1,
    ):
        self.module = TemporalAutoencoderModule(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs
        self.seq_len = seq_len
        self.seq_step = seq_step

    def fit(self, data: np.ndarray):
        sequences = self.create_sequences(data, self.seq_len, self.seq_step)
        dataset = TensorDataset(torch.Tensor(sequences))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self.module, dataloader)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        batch_size, seq_len, input_size = data.shape
        x_hat = self.module(torch.Tensor(data)).detach()
        rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=2)
        rec_error = rec_error.reshape((batch_size, seq_len, 1))

        binary_predictions = (rec_error > self.threshold).astype(int)
        return binary_predictions, rec_error

    @staticmethod
    def create_sequences(data: np.ndarray, seq_len: int, step: int = 1) -> np.ndarray:
        sequences = []
        for i in range(0, len(data) - seq_len + 1, step):
            sequences.append(data[i : i + seq_len])
        return np.stack(sequences)

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
        seq_len: int = 10,
        seq_step: int = 1,
    ):
        self.module = VariationalTemporalAutoencoderModule(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs
        self.seq_len = seq_len
        self.seq_step = seq_step

    def fit(self, data: np.ndarray):
        sequences = self.create_sequences(data, self.seq_len, self.seq_step)
        dataset = TensorDataset(torch.Tensor(sequences))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self.module, dataloader)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        batch_size, seq_len, input_size = data.shape
        x_hat, mean, var = self.module(torch.Tensor(data))
        x_hat = x_hat.detach()
        rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=2)
        rec_error = rec_error.reshape((batch_size, seq_len, 1))

        binary_predictions = (rec_error > self.threshold).astype(int)
        return binary_predictions, rec_error

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
