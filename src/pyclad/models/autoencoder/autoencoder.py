from typing import Optional, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import TensorDataset

from pyclad.models.autoencoder.loss import VariationalMSELoss
from pyclad.models.model import Model
from pyclad.models.neural_model import lightning_trainer_device_kwargs, resolve_torch_device


class _BaseAutoencoder(Model):
    module_cls: Type["_BaseAutoencoderModule"]
    shuffle: bool = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-2,
        threshold: float = 0.5,
        epochs: int = 20,
        device: Union[str, torch.device, None] = "auto",
    ):
        self.module = self.module_cls(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs
        self.device = resolve_torch_device(device)

    def fit(self, data: np.ndarray):
        self.module.to(self.device)
        dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=self.shuffle)
        trainer = pl.Trainer(max_epochs=self.epochs, **lightning_trainer_device_kwargs(self.device))
        trainer.fit(self.module, dataloader)
        self.module.to(self.device)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        data = np.asarray(data, dtype=np.float32)
        x_hat = self._reconstruct(data)
        rec_error = self._reconstruction_error(data, x_hat)
        return (rec_error > self.threshold).astype(int), rec_error

    def module_getter(self):
        return self.module

    def prepare_data(self, data: np.ndarray, fit=False):
        del fit
        return torch.as_tensor(np.asarray(data, dtype=np.float32), dtype=torch.float32)

    def compute_loss(self, module: nn.Module, batch: torch.Tensor):
        return module.batch_loss(batch)

    def forward_batch(self, batch: torch.Tensor, apply_masking=False):
        del apply_masking
        output = self.module(batch)
        x_hat = self.module.reconstruction_from_output(output)
        z = self.encode_batch(batch)
        return x_hat, z, None

    def encode_batch(self, batch: torch.Tensor):
        return self.module.encode_batch(batch)

    def name(self) -> str:
        return self.__class__.__name__

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "lr": self.module.lr,
            "epochs": self.epochs,
            "device": str(self.device),
        }

    def _reconstruct(self, data: np.ndarray) -> np.ndarray:
        self.module.to(self.device)
        self.module.eval()
        with torch.no_grad():
            tensor_data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
            output = self.module(tensor_data)
            return self.module.reconstruction_from_output(output).detach().cpu().numpy()

    def _reconstruction_error(self, data: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        return ((data - x_hat) ** 2).mean(axis=1)


class Autoencoder(_BaseAutoencoder):
    shuffle = True


class TemporalAutoencoder(_BaseAutoencoder):
    shuffle = False

    def _reconstruction_error(self, data: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = data.shape
        return ((data - x_hat) ** 2).mean(axis=2).reshape((batch_size, seq_len, 1))


class VariationalTemporalAutoencoder(TemporalAutoencoder):
    @staticmethod
    def create_sequences(data: np.ndarray, seq_len: int, step: int = 1) -> np.ndarray:
        return np.stack([data[i : i + seq_len] for i in range(0, len(data) - seq_len + 1, step)])


class _BaseAutoencoderModule(pl.LightningModule):
    loss_cls = nn.MSELoss

    def __init__(self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.train_loss = self.loss_cls()
        self.val_loss = self.loss_cls()

    def training_step(self, batch, batch_idx):
        loss = self.batch_loss(batch[0], self.train_loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.batch_loss(batch[0], self.val_loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def batch_loss(self, x: torch.Tensor, loss_fn: Optional[nn.Module] = None):
        loss_fn = self.train_loss if loss_fn is None else loss_fn
        return loss_fn(self(x), x)

    def encode_batch(self, x: torch.Tensor):
        return self.encoder(x)

    @staticmethod
    def reconstruction_from_output(output):
        return output[0] if isinstance(output, (tuple, list)) else output


class AutoencoderModule(_BaseAutoencoderModule):
    def forward(self, x):
        return self.decoder(self.encoder(x))


class TemporalAutoencoderModule(AutoencoderModule):
    pass


class VariationalTemporalAutoencoderModule(_BaseAutoencoderModule):
    loss_cls = VariationalMSELoss

    def forward(self, x):
        mean, var = self.encoder(x)
        return self.decoder(self.reparametrize(mean, var)), mean, var

    @staticmethod
    def reparametrize(mean, var):
        return mean + var * torch.randn_like(var)

    def batch_loss(self, x: torch.Tensor, loss_fn: Optional[nn.Module] = None):
        loss_fn = self.train_loss if loss_fn is None else loss_fn
        x_hat, mean, var = self(x)
        return loss_fn(x_hat, x, mean, var)

    def encode_batch(self, x: torch.Tensor):
        return self.encoder(x)[0]


Autoencoder.module_cls = AutoencoderModule
TemporalAutoencoder.module_cls = TemporalAutoencoderModule
VariationalTemporalAutoencoder.module_cls = VariationalTemporalAutoencoderModule
