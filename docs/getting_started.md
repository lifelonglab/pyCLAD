# Your first experiment

This page walks through one complete pyCLAD experiment, step by step. It trains an Isolation Forest on three concepts,
one after another, and measures how much the model forgets. The [full script](#full-script) is at the end of the page;
it runs in a few seconds and only needs `pip install pyclad`.

If a term is new to you, [Core concepts](learn/concepts.md) explains it in more detail.

## The building blocks

An experiment combines five things:

- **Dataset**: a sequence of *concepts*. A concept is one distribution, activity or task, with training data and
  labelled test data.
- **Model**: an anomaly detector, for example an Isolation Forest.
- **Strategy**: decides how the model is updated as new concepts arrive, so that it adapts to them without forgetting
  the old ones.
- **Scenario**: the loop that feeds concepts to the strategy, and decides what the strategy is told about them.
- **Callbacks**: hooks that the scenario calls along the way, for example to compute metrics or measure time.

## 1. Make the run repeatable

Call `set_seed` once, before anything else. It seeds Python's `random`, NumPy and PyTorch (when installed), and
returns an object that saves the seed with the results:

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:seed"
```

## 2. Prepare a dataset

Each concept has a *name*, *data* and, for test concepts, *labels*: `0` for a normal sample and `1` for an anomaly.
Here we generate three concepts, centred at 0, 10 and 20. Training data is normal only; each test concept also holds 10
anomalies:

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:concepts"
```

A `ConceptsDataset` groups the concepts. The order of `train_concepts` is the order in which the strategy learns
them:

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:dataset"
```

To use your own data, split it into concepts in the same way. pyCLAD also ships ready-to-use benchmarks, which are
downloaded on first use; see [Datasets](datasets.md).

## 3. Choose a model and a strategy

A strategy wraps a model and decides how it is updated. `NaiveStrategy` simply retrains the model on each new concept,
with no protection against forgetting, which makes it the usual lower bound in a comparison:

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:strategy"
```

[Strategies](strategies.md) lists the other strategies, such as replay and regularization methods, and
[Models](models.md) lists the models.

## 4. Define what to measure

After each concept is learned, the scenario evaluates the strategy on **every** test concept: the ones already learned,
the current one and the future ones. `ConceptMetricCallback` collects these results in a matrix $R$, where $R_{i,j}$
is the score on concept $j$ after learning concept $i$, and summarizes it with continual metrics:

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:callbacks"
```

- `base_metric` scores one evaluated concept. ROC-AUC uses the anomaly scores, so it does not depend on a
  threshold.
- `summarized_metrics` turn the whole matrix into one number each. For example, `BackwardTransfer` measures how
  learning new concepts changed the results on earlier ones.
- `stepwise_metrics` report one value after each learned concept, such as `ForgettingMeasure`.

`TimeEvaluationCallback` measures the training and evaluation time. See [Metrics](metrics.md) and
[Callbacks](callbacks.md) for the others.

## 5. Run the scenario

The scenario decides what the strategy is told about the stream. In a *concept-incremental* scenario, each
`learn()` call is one whole concept, but the strategy is not told which concept it is. See [Scenarios](scenarios.md)
for the other scenarios.

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:scenario"
```

## 6. Save the results

`JsonOutputWriter` saves everything that describes the run: the seed, the model's and strategy's parameters, the
dataset, and the results collected by the callbacks.

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:output"
```

Part of the resulting `output.json`:

```json
{
    "seed": 42,
    "model": {"name": "IsolationForest", "n_estimators": 100, "random_state": 42, ...},
    "dataset": {"name": "GeneratedDataset", "train_concepts_no": 3, "test_concepts_no": 3},
    "strategy": {"name": "Naive"},
    "concept_metric_callback_ROC-AUC": {
        "base_metric_name": "ROC-AUC",
        "metrics": {"ContinualAverage": 0.67, "BackwardTransfer": -0.47, "ForwardTransfer": 0.5},
        "stepwise_metrics": {"ForgettingMeasure": [0.0, 0.6, 0.7]},
        "concepts_order": ["concept1", "concept2", "concept3"],
        "metric_matrix": {
            "concept1": {"concept1": 1.0, "concept2": 0.5, "concept3": 0.5},
            "concept2": {"concept1": 0.4, "concept2": 1.0, "concept3": 0.5},
            "concept3": {"concept1": 0.5, "concept2": 0.1, "concept3": 1.0}
        },
        ...
    },
    "time_evaluation_callback": {...}
}
```

## 7. Read the results

The matrix shows what happened. Each concept is detected perfectly right after it is learned (the diagonal is 1.0),
but once the model moves on to the next concept, the score on the earlier one drops to chance level or below. This is
*catastrophic forgetting*, and it is what continual strategies try to prevent. The negative `BackwardTransfer` and the
growing `ForgettingMeasure` summarize the same thing.

Always take numbers from the JSON file, and plot the matrix to see it at a glance:

```python
--8<-- "docs/snippets/getting_started/first_experiment.py:results"
```

## Next steps

- Replace `NaiveStrategy` with `CumulativeStrategy` or a replay strategy and compare the results. A fair comparison
  uses the same dataset, concept order, scenario and metrics, and several seeds.
- Learn the [core concepts](learn/concepts.md) behind scenarios, strategies and metrics.
- Try a real benchmark from [Datasets](datasets.md), or one of the [examples](examples.md).

## Full script

```python linenums="1"
--8<-- "docs/snippets/getting_started/first_experiment.py"
```
