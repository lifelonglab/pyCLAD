import numpy as np
import torch
from pyod.models.deep_svdd import DeepSVDD, InnerDeepSVDD, optimizer_dict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from pyclad.models.neural_model import NeuralModel


class ContinualDeepSVDD(NeuralModel, DeepSVDD):
    """DeepSVDD adapted for continual fine-tuning, with upstream bugs fixed.

    Changes vs PyOD's DeepSVDD:
    - loss.backward() was commented out (no training occurred) — fixed.
    - Center c was always 0.0 (stored on model_.c but read from self.c) — fixed.
    - w_d was a pre-loop constant so contributed nothing to gradients — removed.
    - Model is preserved across fit() calls; center is re-estimated from the
      current network on each new concept's data, enabling warm-start fine-tuning.
    """

    def set_input_shape(self, X):
        self.n_samples_, self.n_features_ = X.shape

    def prepare_preprocessing(self, fit_data):
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(fit_data)

    def ensure_module(self, n_features):
        del n_features
        if np.min(self.hidden_neurons) > self.n_features_ and self.use_ae:
            raise ValueError("The number of neurons should not exceed the number of features")

        if self.model_ is None:
            self.model_ = InnerDeepSVDD(
                self.n_features,
                use_ae=self.use_ae,
                hidden_neurons=self.hidden_neurons,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                dropout_rate=self.dropout_rate,
                l2_regularizer=self.l2_regularizer,
            )

    def after_prepare_fit(self, X, fit_data):
        del X
        fit_norm = self.transform_prepared_data(fit_data)
        device = next(self.model_.parameters()).device
        fit_tensor = torch.as_tensor(np.random.permutation(fit_norm), dtype=torch.float32, device=device)
        self.model_._init_c(fit_tensor)
        self.model_.c = self.model_.c.detach()
        self.c = self.model_.c

    def fit_dataset(self, X, y=None):
        del y
        X_tensor = self.prepare_data(X)
        return TensorDataset(X_tensor, X_tensor)

    def train_fit_loader(self, train_loader):
        optimizer = optimizer_dict[self.optimizer](self.model_.parameters(), weight_decay=self.l2_regularizer)

        best_loss = float("inf")
        best_model_dict = None

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                outputs, representation = self.forward_with_representation(self.model_, batch_x)
                dist = torch.sum((representation - self.c) ** 2, dim=-1)
                if self.use_ae:
                    loss = torch.mean(dist) + torch.mean(torch.square(outputs - batch_x))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_dict = self.model_.state_dict()
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}")

        self.best_model_dict = best_model_dict

    def compute_loss(self, module, batch):
        outputs, representation = self.forward_with_representation(module, batch)
        center = torch.as_tensor(self.c, dtype=representation.dtype, device=representation.device)
        dist = torch.sum((representation - center) ** 2, dim=-1)
        loss = torch.mean(dist)
        if self.use_ae:
            loss = loss + torch.mean(torch.square(outputs - batch))
        return loss

    def snapshot_model_state(self):
        self.best_model_dict = self.clone_state_dict(self.model_.state_dict())
        return self.best_model_dict

    def decision_function(self, X):
        X_tensor = self.prepare_data(X)
        device = next(self.model_.parameters()).device
        X_tensor = X_tensor.to(device)
        self.model_.eval()
        with torch.no_grad():
            _, representation = self.forward_with_representation(self.model_, X_tensor)
            center = torch.as_tensor(self.c, dtype=representation.dtype, device=representation.device)
            dist = torch.sum((representation - center) ** 2, dim=-1)
        return dist.detach().cpu().numpy()

    @staticmethod
    def forward_with_representation(module, batch):
        intermediate_output = {}
        net_output = module.model._modules.get("net_output")
        if net_output is None:
            output = module(batch)
            return output, output

        hook_handle = net_output.register_forward_hook(
            lambda _module, _inputs, output: intermediate_output.update({"net_output": output})
        )
        try:
            output = module(batch)
        finally:
            hook_handle.remove()
        return output, intermediate_output["net_output"]
