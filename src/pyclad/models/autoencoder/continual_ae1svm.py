import numpy as np
import torch
from pyod.models.ae1svm import AE1SVM, InnerAE1SVM, TorchDataset
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from sklearn.utils.validation import check_array

from pyclad.models.neural_model import NeuralModel


class ContinualAE1SVM(NeuralModel, AE1SVM):
    """AE1SVM adapted for continual fine-tuning.

    Identical to PyOD's AE1SVM except the network is preserved across fit()
    calls, enabling warm-start fine-tuning on new concepts. The preprocessing
    scaler and optimizer are still reset each call (adapt to new data statistics;
    fresh Adam state).
    """

    def set_input_shape(self, X):
        self.n_samples_, self.n_features_ = X.shape

    def prepare_preprocessing(self, fit_data):
        if self.preprocessing:
            self.mean = np.mean(fit_data, axis=0)
            self.std = self.safe_std(np.std(fit_data, axis=0))

    def ensure_module(self, n_features):
        if self.model is None:
            self.model = InnerAE1SVM(
                n_features=n_features,
                encoding_dim=32,
                rff_dim=self.kernel_approx_features,
                sigma=self.sigma,
                hidden_neurons=self.hidden_neurons,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                hidden_activation=self.hidden_activation,
            ).to(self.device)
        else:
            self.model = self.model.to(self.device)

    def fit_dataset(self, X, y=None):
        del y
        if self.preprocessing:
            return TorchDataset(X=X, mean=self.mean, std=self.std, return_idx=True)
        return TorchDataset(X=X, return_idx=True)

    def train_fit_loader(self, train_loader):
        self._train_autoencoder(train_loader)

        if self.best_model_dict is not None:
            self.model.load_state_dict(self.best_model_dict)
        else:
            raise ValueError("Training failed, no valid model state found")

    def fit_drop_last(self):
        return bool(self.batch_norm)

    def compute_loss(self, module, batch):
        reconstructions, rff_features = module(batch)
        recon_loss = self.loss_fn(batch, reconstructions)
        svm_scores = module.svm_decision_function(rff_features)
        svm_loss = torch.mean(torch.clamp(1 - svm_scores, min=0))
        return self.alpha * recon_loss + svm_loss

    def snapshot_model_state(self):
        return self.clone_state_dict(self.model.state_dict())

    def drop_last(self):
        return bool(self.batch_norm)

    def decision_function(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["model", "best_model_dict"])
        X = check_array(X)
        dataset = (
            TorchDataset(X=X, mean=self.mean, std=self.std, return_idx=True)
            if self.preprocessing
            else TorchDataset(X=X, return_idx=True)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        outlier_scores = np.zeros(X.shape[0])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data = data.to(self.device).float()
                reconstructions, _ = self.model(data)
                scores = pairwise_distances_no_broadcast(data.cpu().numpy(), reconstructions.cpu().numpy())
                outlier_scores[data_idx.cpu().numpy()] = scores  # fix: tensor → numpy index
        return outlier_scores
