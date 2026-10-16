# Models

### Overview

A model is an anomaly detector. Strategies decide which data a model is trained on; the model does the detection.
Every model implements the `Model` contract (`pyclad.models.model`):

- `fit(data)` trains the model on an array whose first axis runs over samples.
- `predict(data)` returns a `PredictionResults` with two arrays, one entry per sample:
    - `anomaly_scores`: **a higher score means a more anomalous sample**. The scale is up to the model; ranking
      metrics such as ROC-AUC only use the order.
    - `y_pred`: **`0` for a normal sample, `1` for an anomaly**.
- `name()` and, optionally, `additional_info()` describe the model in the output JSON. Return every hyperparameter
  that shapes the result.

`fit` only ever sees training data. `predict` must not change what the model has learned.

Models that train on labels implement `SupervisedModel` instead, whose `fit` takes `data` and `labels`. It is a
separate contract, not a subclass of `Model`, and is used with `SupervisedStrategy` and
`SupervisedConceptIncrementalScenario`.

### PyOD models

pyCLAD ships adapters for several [PyOD](https://pyod.readthedocs.io/) detectors: `IsolationForestAdapter` [1],
`LocalOutlierFactorAdapter` [2], `OneClassSVMAdapter`, `COPODAdapter` and `ECODAdapter`, all in
`pyclad.models.adapters.pyod_adapters`. Keyword arguments go to the PyOD detector, and `additional_info()` reports all
of its parameters. Any other PyOD detector can be wrapped with `PyODAdapter`:

```python
--8<-- "docs/snippets/models/pyod_models.py:pyod"
```

```python
--8<-- "docs/snippets/models/pyod_models.py:predict"
```

PyOD already follows pyCLAD's conventions (higher score = more anomalous, `1` = anomaly), so the adapter passes its
results through unchanged.

### Writing your own model

Subclass `Model` and convert to pyCLAD's conventions in `predict`. [Implementing your own model](guides/custom_model.md)
walks through wrapping scikit-learn's Isolation Forest, which uses the opposite conventions, and writing a
`TorchBackbone`, with a checklist.

### PyTorch models

PyTorch models (`pip install "pyclad[torch]"`) are split in two parts:

- A **backbone** (`TorchBackbone`) holds the network, its loss and its optimizer. It implements `get_module()`,
  `get_optimizer()`, `compute_loss(x)`, `forward(x)` and `name()`. It also needs `predict(data)` returning
  `PredictionResults`: the base class does not declare it, but `TorchModelAdapter` and the strategies call it.
  Strategies that change the training itself, such as EWC, LwF, DER++ and A-GEM, work directly with a backbone.
- A **runner** (`TorchRunner`) owns the training loop: epochs, an optional validation split and early stopping.
  `StandardRunner(max_epochs, validation_fraction=0.0, early_stopping=None, seed=None)` is the standard one. The
  validation set is always split off the training data of the current concept, never taken from test data.

`TorchModelAdapter(backbone, runner, batch_size)` combines the two into a `Model`, so a PyTorch model also works with
the strategies that take any model (Naive, Cumulative, replay, VLAD):

```python
--8<-- "docs/snippets/models/torch_models.py:autoencoder"
```

`EarlyStopping(patience, min_delta, restore_best_weights)` stops training when the validation loss has not improved
for `patience` epochs and, by default, restores the best weights. The runner resets it at the start of each concept,
so each concept is judged on its own.

#### Autoencoders

Autoencoders score a sample by its reconstruction error:

- `Autoencoder(encoder, decoder, lr, threshold)` takes any encoder and decoder `nn.Module`.
- `VariationalAutoencoder(encoder, decoder, hidden_dim, latent_dim, lr, threshold, kl_weight)` adds the mean and
  log-variance heads between the encoder (which outputs `hidden_dim` features) and the decoder (which takes
  `latent_dim` features). It scores with the latent mean, so scores do not vary between calls.

```python
--8<-- "docs/snippets/models/torch_models.py:vae"
```

!!! warning "Fixed threshold"
    The autoencoders turn reconstruction errors into `y_pred` with a fixed `threshold` (0.5 by default), which is not
    calibrated to the data. Threshold-based metrics such as F1 are therefore not meaningful for them unless you set a
    threshold derived from training data. Ranking metrics (ROC-AUC, average precision) are unaffected.

For time series, `TemporalAutoencoder` and `VariationalTemporalAutoencoder` reconstruct windows of shape
`(n_windows, seq_len, n_features)` with GRU, LSTM or TCN encoders and decoders (`GRUEncoder`, `LSTMEncoder`,
`TCNEncoder` and their decoders in `pyclad.models.autoencoder.standard`, and their variational counterparts in
`pyclad.models.autoencoder.variational`). They are trained with
PyTorch Lightning for a fixed number of `epochs`. See the `*_tae_torch_example.py` scripts in
[examples/models](https://github.com/lifelonglab/pyCLAD/tree/main/examples/models).

#### CARLA

`Carla(backbone, ...)` (`pyclad.models.contrastive.carla`) is a self-supervised contrastive model for time-series
anomaly detection [3]. It takes a feature extractor such as `ResNet1D` and trains in two stages, a triplet-loss pretext
stage with injected anomalies and a self-supervised classification stage. See
[carla_energy_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/models/carla_energy_example.py).

### Time series with tabular models

`convert_dataset_to_overlapping_windows` (in `pyclad.data`) turns every concept of a dataset into overlapping windows.
`FlattenTimeSeriesAdapter` then lets any model that expects 2D input score them:

```python
--8<-- "docs/snippets/models/time_series.py:windows"
```

### Image models

Image anomaly detection models (PatchCore, FastFlow, RD4AD, PaSTe, UCAD and others) are described in
[Vision](vision.md).

### References

[1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE.

[2] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data* (pp. 93-104).

[3] Darban, Z. Z., Webb, G. I., Pan, S., Aggarwal, C. C., & Salehi, M. CARLA: Self-supervised contrastive representation learning for time series anomaly detection. [arXiv:2308.09296](https://arxiv.org/abs/2308.09296).
