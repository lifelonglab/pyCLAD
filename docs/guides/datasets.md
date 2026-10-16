# Using built-in datasets and step schedules

This guide goes through running an experiment on a benchmark: loading it, checking it, choosing an order and, if
needed, grouping concepts into larger training steps. [Datasets](../datasets.md) lists the benchmarks and the
formats in detail.

## 1. Load the benchmark

Built-in benchmarks download from Hugging Face on first use and are cached afterwards; pass `cache_dir` to choose
where, for example a shared disk on a cluster. Most take a `dataset_type`, and the tabular CAD benchmarks an
`ordering`. "Built-in Datasets" in [Datasets](../datasets.md#built-in-datasets) lists them, with code.

## 2. Check the dataset before running anything

Look at how many samples each concept has and how many test samples are anomalies. A test concept with a single
class makes ROC-AUC and average precision undefined for it, which stops the run by default (see "Undefined values"
in [Metrics](../metrics.md#undefined-values)). Better to know before a long run:

```python
--8<-- "docs/snippets/guides/inspect_dataset.py:inspect"
```

```
Cluster_0    train=   600  test=   800  anomalies= 50.0%
Cluster_1    train=   600  test=   800  anomalies= 50.0%
...
```

## 3. Fix the concept order

The order of the train concepts is the continual order, and results depend on it. Use one of the benchmark's
predefined orderings, or reorder the concepts yourself before building the dataset, and use the same order for every
method you compare. The dataset's `name` includes the ordering of the CAD benchmarks, so it is saved with the results.

## 4. Optionally, group concepts into training steps

Some protocols train on several concepts at once, for example a large base step followed by single new concepts,
while still evaluating each concept on its own. `apply_step_schedule` merges *consecutive* concepts, so apply it
after fixing the order. The metric matrix then has one row per training step and one column per concept, which the
square-only metrics (`ContinualAverage`, `BackwardTransfer`, `ForwardTransfer`, `ForgettingMeasure`) reject: use
`ScheduleAwareConceptMetricCallback` and the schedule-aware metrics instead. The notation and a complete example are
in "Step Schedule (Grouping)" in [Datasets](../datasets.md#step-schedule-grouping), and the metrics in
[Metrics](../metrics.md#rectangular-matrices-and-step-schedules).

## 5. Run and report

Run the scenario as in [your first experiment](../getting_started.md). When reporting, name the benchmark, its
type or ordering, and the step schedule, and cite the benchmark's source as its class docstring asks.
