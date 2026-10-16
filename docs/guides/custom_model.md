# Implementing your own model

A model is an anomaly detector behind the `Model` interface: `fit(data)`, `predict(data)` returning
`PredictionResults`, and `name()`. Any library can be wrapped this way. See [Models](../models.md) for the contract
and the models pyCLAD ships.

## Wrap a detector

Convert to pyCLAD's conventions at the boundary: **higher anomaly score = more anomalous**, and **`0` = normal,
`1` = anomaly**. scikit-learn's Isolation Forest uses the opposite for both:

```python
--8<-- "docs/snippets/models/custom_model.py:model"
```

A PyOD detector needs no class of its own: `PyODAdapter(detector, model_name=...)` already follows the conventions.

## A PyTorch model

To use a PyTorch network, implement `TorchBackbone` instead. The backbone holds the network, its loss and its
optimizer; a runner owns the training loop. Wrapped in `TorchModelAdapter` it works with every strategy, and the
regularization strategies (EWC, LwF, DER++, A-GEM) take it directly:

```python
--8<-- "docs/snippets/guides/custom_backbone.py:backbone"
```

```python
--8<-- "docs/snippets/guides/custom_backbone.py:use"
```

## Test it in a scenario

A test should fail if the model were wrong, not only check that it runs. Run the model through a short scenario once,
in a module-scoped fixture, and assert on the result:

```python
--8<-- "docs/snippets/guides/custom_model_tests.py:test"
```

## Checklist

- [ ] `predict` returns `PredictionResults`; higher scores mean more anomalous; `y_pred` uses `0`/`1`.
- [ ] `fit` uses only the data it is given. Thresholds and normalisation statistics come from that training data,
      never from test data.
- [ ] `predict` does not change what the model learned.
- [ ] `additional_info()` returns every setting that shapes the results: hyperparameters, backbone name, seed.
- [ ] Randomness comes from a `seed` or `rng` argument and, when none is given, from `pyclad.seed.new_generator()`,
      so `set_seed` makes runs repeatable.
- [ ] Heavy optional packages (torch, a model library) are imported where they are used, not when `pyclad` is
      imported, and are listed in the right extra in `pyproject.toml`.
- [ ] A port of a published model cites the paper and lists every difference from it in the docs.
- [ ] To contribute it to pyCLAD: a test under `tests/models/`, an example in `examples/models/`, and an entry in
      [Models](../models.md). See [Contributing](../contribution.md).
