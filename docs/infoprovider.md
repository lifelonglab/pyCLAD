# Output

pyCLAD saves everything that describes a run (the seed, the dataset, the model, the strategy and the results of the
callbacks) in one JSON file. The file should be enough to tell two runs apart, and the numbers you report should come
from it.

### InfoProvider

`InfoProvider` (`pyclad.output.output_writer`) is implemented by everything that can be saved: datasets, models,
strategies, callbacks and the seed. Its one method, `info()`, returns a dictionary with one top-level key:

| Provider | Key | Content |
|---|---|---|
| `SeedInfo` (returned by `set_seed`) | `seed` | the seed |
| `Dataset` | `dataset` | `name` and `additional_info()`, for example the number of concepts |
| `Model` | `model` | `name` and `additional_info()`, the model's hyperparameters |
| `Strategy` | `strategy` | `name` and `additional_info()`, for example buffer sizes and the runner's settings |
| `ConceptMetricCallback` | `concept_metric_callback_<base metric>` | metrics, the metric matrix and the concept orders |
| `TimeEvaluationCallback` | `time_evaluation_callback` | training and evaluation time per concept |

Datasets, models and strategies implement `name()` and override `additional_info()` instead of `info()`; their
base classes build the dictionary. When you write one, return from `additional_info()` every setting that shapes the
results: hyperparameters, buffer sizes, backbone names and seeds.

### JsonOutputWriter

`JsonOutputWriter(path).write(providers)` merges the `info()` of all providers into one JSON object and writes it.

- Undefined values (`NaN`) and infinities are written as `null`, since bare `NaN` is not valid JSON.
- NumPy scalars are converted to Python numbers.
- Providers are merged by their top-level key. Two providers with the same key (for example two
  `ConceptMetricCallback`s with the same base metric) overwrite each other, so give each a different base metric.
- A value that JSON cannot represent, such as a NumPy array or an arbitrary object, is written as the string
  `"<not serializable>"` rather than raising an error. Convert such values to lists or strings in `info()`.

Any object can be saved by implementing `InfoProvider`:

```python
--8<-- "docs/snippets/output/output.py:provider"
```

See [Your first experiment](getting_started.md#6-save-the-results) for the output of a complete run, and how to read
it back and plot the metric matrix.

### PredictionResults

`PredictionResults` (`pyclad.output.prediction_results`) is what models and strategies return from `predict`:

- `y_pred`: one label per sample, `0` for normal and `1` for an anomaly;
- `anomaly_scores`: one score per sample, where a higher score means a more anomalous sample.

The scenario passes both arrays, with the true labels, to the callbacks' `after_evaluation`.
