import logging

import numpy as np
import torch
import torch.nn as nn

from pyclad.data.timeseries import convert_to_overlapping_windows
from pyclad.models.autoencoder.autoencoder import (
    TemporalAutoencoder,
    VariationalTemporalAutoencoder,
)
from pyclad.models.autoencoder.builder import build
from pyclad.models.autoencoder.config import (
    ActivationLayerConfig,
    AutoencoderConfig,
    DecoderConfig,
    DropoutLayerConfig,
    EncoderConfig,
    GRULayerConfig,
)
from pyclad.models.autoencoder.standard.gru import GRUDecoder, GRUEncoder
from pyclad.models.autoencoder.variational.gru import (
    GRUVariationalDecoder,
    GRUVariationalEncoder,
)

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    use_variational_ae = True  # Set to True if you want to use Variational Autoencoder

    batch_size, seq_len = 32, 10
    time_steps, n_features = 5000, 5
    dataset, _ = convert_to_overlapping_windows(seq_len, np.random.rand(time_steps, n_features))

    config = AutoencoderConfig(
        encoder=EncoderConfig(
            layers=[
                GRULayerConfig(
                    kwargs={"input_size": n_features, "hidden_size": 24, "num_layers": 1, "batch_first": True}
                ),
                ActivationLayerConfig(cls=nn.Tanh),
                DropoutLayerConfig(kwargs={"p": 0.1}),
                GRULayerConfig(kwargs={"input_size": 24, "hidden_size": 12, "num_layers": 1, "batch_first": True}),
            ]
        ),
        decoder=DecoderConfig(
            layers=[
                GRULayerConfig(kwargs={"input_size": 12, "hidden_size": 24, "num_layers": 1, "batch_first": True}),
                ActivationLayerConfig(cls=nn.Tanh),
                DropoutLayerConfig(kwargs={"p": 0.1}),
                GRULayerConfig(
                    kwargs={"input_size": 24, "hidden_size": n_features, "num_layers": 1, "batch_first": True}
                ),
                ActivationLayerConfig(cls=nn.Tanh),
                DropoutLayerConfig(kwargs={"p": 0.1}),
            ]
        ),
    )

    _encoder_layers, _decoder_layers = build(config=config)

    if use_variational_ae:
        encoder = GRUVariationalEncoder(_encoder_layers)
        decoder = GRUVariationalDecoder(_decoder_layers, seq_len=seq_len)
        autoencoder = VariationalTemporalAutoencoder(encoder=encoder, decoder=decoder, epochs=5)
    else:
        encoder = GRUEncoder(_encoder_layers)
        decoder = GRUDecoder(_decoder_layers, seq_len=seq_len)
        autoencoder = TemporalAutoencoder(encoder=encoder, decoder=decoder, epochs=5)

    autoencoder.fit(dataset)

    input_batch = torch.randn(batch_size, seq_len, n_features)
    true_labels_batch = torch.randn(batch_size, seq_len, 1)

    binary_predictions, rec_error = autoencoder.predict(input_batch.numpy())
    assert binary_predictions.shape == true_labels_batch.shape
