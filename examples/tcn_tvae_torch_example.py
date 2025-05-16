import logging

import numpy as np
import torch
import torch.nn as nn

from pyclad.models.autoencoder.autoencoder import VariationalTemporalAutoencoder
from pyclad.models.autoencoder.builder import build
from pyclad.models.autoencoder.config import (
    ActivationLayerConfig,
    AutoencoderConfig,
    Conv1dLayerConfig,
    ConvTranspose1dLayerConfig,
    DecoderConfig,
    EncoderConfig,
)
from pyclad.models.autoencoder.variational.tcn import (
    TCNVariationalDecoder,
    TCNVariationalEncoder,
)

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    batch_size, seq_len = 32, 10
    time_steps, n_features = 5000, 5
    dataset = np.random.rand(time_steps, n_features)

    config = AutoencoderConfig(
        seq_len=seq_len,
        encoder=EncoderConfig(
            layers=[
                Conv1dLayerConfig(
                    kwargs={"in_channels": n_features, "out_channels": 24, "kernel_size": 3, "stride": 1, "padding": 1}
                ),
                ActivationLayerConfig(cls=nn.Tanh),
                Conv1dLayerConfig(
                    kwargs={"in_channels": 24, "out_channels": 12, "kernel_size": 3, "stride": 1, "padding": 1}
                ),
                ActivationLayerConfig(cls=nn.Tanh),
            ]
        ),
        decoder=DecoderConfig(
            layers=[
                ConvTranspose1dLayerConfig(
                    kwargs={"in_channels": 12, "out_channels": 24, "kernel_size": 3, "stride": 1, "padding": 1}
                ),
                ActivationLayerConfig(cls=nn.Tanh),
                ConvTranspose1dLayerConfig(
                    kwargs={"in_channels": 24, "out_channels": n_features, "kernel_size": 3, "stride": 1, "padding": 1}
                ),
                ActivationLayerConfig(cls=nn.Tanh),
            ]
        ),
    )

    _encoder_layers, _decoder_layers = build(config=config)
    encoder = TCNVariationalEncoder(_encoder_layers, seq_len=config.seq_len)
    decoder = TCNVariationalDecoder(_decoder_layers, seq_len=config.seq_len)
    autoencoder = VariationalTemporalAutoencoder(encoder=encoder, decoder=decoder, epochs=5, seq_len=config.seq_len)

    autoencoder.fit(dataset)

    input_batch = torch.randn(batch_size, seq_len, n_features)
    true_labels_batch = torch.randn(batch_size, seq_len, 1)

    binary_predictions, rec_error = autoencoder.predict(input_batch.numpy())
    assert binary_predictions.shape == true_labels_batch.shape
