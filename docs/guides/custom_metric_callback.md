# Adding a metric or a callback

## A base metric

A base metric scores one evaluated concept. Subclass `BaseMetric` and implement `compute(anomaly_scores, y_pred,
y_true)` and `name()`. When the metric cannot be computed for a concept, return `NaN`, never a substitute such as 0
or 0.5, so the result cannot be mistaken for a real score:

```python
--8<-- "docs/snippets/guides/custom_metric.py:base"
```

Use it as the `base_metric` of a `ConceptMetricCallback`. Its `name()` becomes part of the callback's key in the
output, so two base metrics must have different names.

## A continual metric

A continual metric summarizes the matrix $R$, where `metric_matrix[i][j]` is the score on concept `j` after learning
concept `i`:

- `SummarizedMetric.compute(matrix)` returns one value;
- `StepwiseConceptMetric.compute(matrix)` returns one value per learned concept;
- `ScheduleAwareMetric.compute(matrix, first_seen_steps)` works on the rectangular matrices of step schedules.

A metric that reads the diagonal or the triangles only makes sense on a square matrix, so it calls
`validate_square_matrix` first. A cell that is `NaN` makes the result `NaN`; averaging through
`pyclad.metrics.continual.concepts_metric.mean_or_nan` does this for you:

```python
--8<-- "docs/snippets/guides/custom_metric.py:summarized"
```

## Test it on a matrix you computed by hand

The test should contain the expected number, worked out by hand, so that it fails if the definition is
implemented wrong. Cover the undefined case too:

```python
--8<-- "docs/snippets/guides/custom_metric.py:tests"
```

## A callback

Callbacks observe a run through hooks such as `after_training` and `after_evaluation`. Subclass `Callback`, accept
`*args, **kwargs` in every hook, and implement `InfoProvider.info()` to save results:

```python
--8<-- "docs/snippets/callbacks/callbacks.py:custom"
```

Release any resource in `after_scenario`, which runs even when the scenario fails. See [Callbacks](../callbacks.md)
for every hook and its arguments.

## Checklist

- [ ] Undefined values are `NaN`, and a `NaN` input makes the result `NaN`.
- [ ] Square-only metrics call `validate_square_matrix`.
- [ ] The test checks a hand-computed value, and the undefined case.
- [ ] `name()` is unique among the metrics used together.
- [ ] A metric that changes the meaning of an existing one gets a new class with a new name.
- [ ] To contribute it: a test under `tests/metrics/` or `tests/callbacks/`, and an entry in
      [Metrics](../metrics.md) or [Callbacks](../callbacks.md).
