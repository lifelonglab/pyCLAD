import numpy as np
import pytest
import torch
from torch import nn

from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.autoencoder.autoencoder import (
    VariationalAutoencoder,
    VariationalAutoencoderModule,
)
from pyclad.models.training.runners.standard import StandardRunner

FEATURES, HIDDEN_DIM, LATENT_DIM = 6, 4, 2


def _encoder() -> nn.Module:
    return nn.Sequential(nn.Linear(FEATURES, HIDDEN_DIM), nn.ReLU())


def _decoder() -> nn.Module:
    return nn.Sequential(nn.Linear(LATENT_DIM, FEATURES))


def _backbone(**kwargs) -> VariationalAutoencoder:
    return VariationalAutoencoder(_encoder(), _decoder(), hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, **kwargs)


def _data(n: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, FEATURES)).astype(np.float32)


# -- module ------------------------------------------------------------------


def test_encode_returns_mean_and_logvar_of_latent_dim():
    module = VariationalAutoencoderModule(_encoder(), _decoder(), HIDDEN_DIM, LATENT_DIM)
    mean, log_var = module.encode(torch.rand(8, FEATURES))
    assert mean.shape == (8, LATENT_DIM)
    assert log_var.shape == (8, LATENT_DIM)


def test_forward_reconstructs_input_shape_and_returns_latent_params():
    module = VariationalAutoencoderModule(_encoder(), _decoder(), HIDDEN_DIM, LATENT_DIM)
    x = torch.rand(8, FEATURES)
    x_hat, mean, log_var = module(x)
    assert x_hat.shape == x.shape
    assert mean.shape == log_var.shape == (8, LATENT_DIM)


def test_reconstruct_uses_latent_mean_and_is_deterministic():
    module = VariationalAutoencoderModule(_encoder(), _decoder(), HIDDEN_DIM, LATENT_DIM)
    module.eval()
    x = torch.rand(8, FEATURES)
    # reconstruct() must not sample: repeated calls are identical, and match decoding the mean directly.
    first, second = module.reconstruct(x), module.reconstruct(x)
    assert torch.equal(first, second)
    mean, _ = module.encode(x)
    assert torch.equal(first, module.decoder(mean))


def test_reparametrize_collapses_to_mean_when_variance_is_tiny():
    mean = torch.tensor([[1.0, -2.0]])
    log_var = torch.full((1, LATENT_DIM), -50.0)  # std ~ exp(-25) ~ 0
    z = VariationalAutoencoderModule.reparametrize(mean, log_var)
    assert torch.allclose(z, mean, atol=1e-6)


# -- loss --------------------------------------------------------------------


def test_compute_loss_is_a_differentiable_scalar():
    backbone = _backbone()
    loss = backbone.compute_loss(torch.rand(16, FEATURES))
    assert loss.ndim == 0
    assert loss.requires_grad
    loss.backward()  # must be backward-able per the TorchBackbone contract


def test_kl_weight_scales_the_kl_term():
    x = torch.rand(16, FEATURES)

    # Share one backbone and vary only kl_weight; seed before each call so the
    # reparameterization sample (and thus the reconstruction term) is identical.
    backbone = _backbone(kl_weight=0.0)
    torch.manual_seed(0)
    recon_only = backbone.compute_loss(x).item()

    backbone.kl_weight = 1.0
    torch.manual_seed(0)
    with_kl = backbone.compute_loss(x).item()

    backbone.kl_weight = 2.0
    torch.manual_seed(0)
    with_double_kl = backbone.compute_loss(x).item()

    kl = with_kl - recon_only
    assert kl >= 0.0  # KL divergence is non-negative
    assert with_double_kl - recon_only == pytest.approx(2 * kl, rel=1e-5)


# -- predict -----------------------------------------------------------------


def test_predict_returns_per_sample_scores_and_labels():
    backbone = _backbone()
    data = _data(20)
    result = backbone.predict(data)
    assert result.anomaly_scores.shape == (20,)
    assert result.y_pred.shape == (20,)
    assert set(np.unique(result.y_pred)).issubset({0, 1})


def test_predict_thresholds_anomaly_scores():
    backbone = _backbone(threshold=1e9)  # unreachable threshold -> everything is normal
    result = backbone.predict(_data(20))
    assert np.all(result.y_pred == 0)

    backbone = _backbone(threshold=-1.0)  # everything exceeds a negative threshold -> all anomalies
    result = backbone.predict(_data(20))
    assert np.all(result.y_pred == 1)


def test_predict_is_deterministic():
    backbone = _backbone()
    data = _data(20)
    np.testing.assert_array_equal(backbone.predict(data).anomaly_scores, backbone.predict(data).anomaly_scores)


# -- integration with adapter + runner --------------------------------------


def test_fit_through_adapter_reduces_reconstruction_error():
    backbone = _backbone(lr=1e-2)
    model = TorchModelAdapter(backbone, StandardRunner(max_epochs=50), batch_size=16)
    data = _data(64)

    before = backbone.predict(data).anomaly_scores.mean()
    model.fit(data)
    after = backbone.predict(data).anomaly_scores.mean()

    assert after < before


def test_additional_info_reports_dims_and_hyperparameters():
    info = _backbone(lr=0.003, threshold=0.7, kl_weight=0.5).additional_info()
    assert info["hidden_dim"] == HIDDEN_DIM
    assert info["latent_dim"] == LATENT_DIM
    assert info["lr"] == 0.003
    assert info["threshold"] == 0.7
    assert info["kl_weight"] == 0.5
