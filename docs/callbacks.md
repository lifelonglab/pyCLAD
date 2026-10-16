# Callbacks

### Overview

Callbacks are hooks that a scenario calls at fixed points of a run. pyCLAD uses them to compute metrics and to
measure time, memory and energy, and you can write your own.

A scenario calls these methods of every callback, in this order:

| Method | When | Keyword arguments |
|---|---|---|
| `before_scenario` | once, at the start | |
| `before_concept_processing` | before each train concept | `concept` |
| `before_training` | before `strategy.learn` | |
| `after_training` | after `strategy.learn` | `learned_concept` |
| `before_evaluation` | before each `strategy.predict` | |
| `after_evaluation` | after each `strategy.predict` | `evaluated_concept`, `y_true`, `y_pred`, `anomaly_scores` |
| `after_concept_processing` | after all evaluations for a train concept | `concept` |
| `after_scenario` | once, at the end, **also when the run fails** | |

Callbacks are passed to the scenario as a list and called in list order. A callback that reports results also
implements `InfoProvider`, so its results can be saved with `JsonOutputWriter`.

### Available callbacks

All of them are importable from `pyclad.callbacks`.

- **ConceptMetricCallback**(`base_metric`, `summarized_metrics`, `stepwise_metrics=None`, `on_undefined="raise"`)
  computes `base_metric` for every pair of (learned concept, evaluated concept), builds the metric matrix and reports
  the continual metrics computed from it. Its output, under `concept_metric_callback_<metric name>`, contains the
  summarized and stepwise metrics, the full `metric_matrix`, `concepts_order` (the learning order), `test_order` and
  `undefined_concepts`. `on_undefined` decides what happens when the base metric cannot be computed; see
  "Undefined values" in [Metrics](metrics.md#undefined-values).
- **ScheduleAwareConceptMetricCallback** adds `schedule_aware_metrics` and `first_seen_step`, for runs whose training
  steps group several concepts; see [Metrics](metrics.md#rectangular-matrices-and-step-schedules).
- **GroupedConceptMetricCallback**(`base_metric`, `group_by_concept`, `summarized_metrics`, `on_undefined`) reports
  the base metric averaged per group of concepts, where `group_by_concept` maps each evaluated concept to its group.
- **TimeEvaluationCallback** measures the training and evaluation time per concept and in total, in seconds.
- **MemoryUsageCallback** records, after each concept, the current and peak memory in bytes, as measured by Python's
  `tracemalloc`. `tracemalloc` sees memory allocated through Python, including NumPy arrays, but not memory that
  native libraries such as PyTorch allocate themselves, nor GPU memory.

    !!! warning "Do not measure time and memory in the same run"
        `tracemalloc` slows Python down considerably while it is active, so a `TimeEvaluationCallback` in the same run
        reports inflated times. Measure time and memory in separate runs.

- **EnergyEvaluationCallback** measures the energy consumption and CO2 emissions of the whole scenario with
  [CodeCarbon](https://codecarbon.io/). **OfflineEnergyEvaluationCallback**(`country_iso_code`, ...) does the same
  without internet access, for the country (and optionally region or cloud provider) you specify.

```python
--8<-- "docs/snippets/callbacks/callbacks.py:builtin"
```

### Writing a callback

Subclass `Callback`, override the hooks you need, and accept `*args, **kwargs` in each of them, since a scenario may
pass more keyword arguments than you use. Release any resource in `after_scenario`. A callback only observes the run:
it must not change the strategy or the data. See [Adding a metric or a callback](guides/custom_metric_callback.md#a-callback)
for an example.
