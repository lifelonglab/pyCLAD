import numpy as np
from pyod.models.vae import VAE, VAEModel
from pyod.utils.torch_utility import TorchDataset

from pyclad.models.neural_model import NeuralModel


class ContinualVAE(NeuralModel, VAE):
    """VAE that warm-starts from existing weights on repeated fit() calls.

    Identical to PyOD's VAE except build_model() skips model creation if a
    model is already present, enabling continual fine-tuning across concepts.
    The optimizer is still reset each call (fresh Adam state), but the network
    weights carry over.
    """

    def build_model(self):
        if getattr(self, "model", None) is not None:
            return
        self.model = VAEModel(
            self.feature_size,
            encoder_neuron_list=self.encoder_neuron_list,
            decoder_neuron_list=self.decoder_neuron_list,
            latent_dim=self.latent_dim,
            hidden_activation_name=self.hidden_activation_name,
            output_activation_name=self.output_activation_name,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
        )

    def set_input_shape(self, X):
        self.data_num, self.feature_size = X.shape

    def prepare_preprocessing(self, fit_data):
        if self.preprocessing:
            self.X_mean = np.mean(fit_data, axis=0)
            self.X_std = self.safe_std(np.std(fit_data, axis=0))

    def ensure_module(self, n_features):
        del n_features
        self.build_model()
        self.model = self.model.to(self.device)

    def fit_dataset(self, X, y=None):
        if self.preprocessing:
            return TorchDataset(X=X, y=y, mean=self.X_mean, std=self.X_std)
        return TorchDataset(X=X, y=y)

    def train_fit_loader(self, train_loader):
        self.training_prepare()
        self.train(train_loader)

    def fit_drop_last(self):
        return True

    def compute_loss(self, module, batch):
        x_recon, z_mu, z_logvar = module(batch)
        return self.criterion(batch, x_recon, z_mu, z_logvar, beta=self.beta, capacity=self.capacity)

    def snapshot_model_state(self):
        return None

    def name(self) -> str:
        return "Continual VAE"
