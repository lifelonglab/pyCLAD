import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import TensorDataset

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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.val_loss(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
