# Running a reproducible comparison

A number in a paper should mean the same thing to every reader, and someone else should get it again from your code.
This guide shows how to compare methods with pyCLAD so that both hold.

## Hold everything fixed except the method

Compare methods on the same:

- dataset, and the same **concept order** (built-in benchmarks offer several orderings, and the order changes the
  results);
- scenario;
- step schedule, if you group concepts;
- base metric and continual metrics;
- model and model settings, when you compare strategies.

State each of them next to the numbers.

## Run several seeds

One seed is one sample. Call `set_seed` at the start of every run, before creating the data, the model and the
strategy, and pass what it returns to the output writer, so the seed is saved with the results. Run each method with
several seeds and report the mean and the spread.

The script below compares three strategies over three seeds and saves one JSON file per run:

```python
--8<-- "docs/snippets/guides/comparison.py:run"
```

`on_undefined="propagate"` lets a run finish when a metric is undefined for some concept, and records `NaN` instead
of stopping. Whether you use it or not, every run you started belongs in the results.

## Take the numbers from the saved results

Aggregate the JSON files, not numbers copied from logs or from memory, and keep the files with your results:

```python
--8<-- "docs/snippets/guides/comparison.py:aggregate"
```

```
strategy    metric                  mean     std  runs
Naive       ContinualAverage       0.589   0.008  3 (0 NaN)
Naive       BackwardTransfer      -0.500   0.001  3 (0 NaN)
Replay      ContinualAverage       0.943   0.021  3 (0 NaN)
Replay      BackwardTransfer      -0.099   0.037  3 (0 NaN)
Cumulative  ContinualAverage       0.975   0.006  3 (0 NaN)
Cumulative  BackwardTransfer      -0.042   0.006  3 (0 NaN)
```

A `NaN` mean says that at least one run had an undefined value. Report it and the number of such runs; do not drop
the run or rerun it with another seed.

## Do not tune on the test concepts

Hyperparameters, thresholds, early stopping and the choice of epoch must come from training data, or from a
validation split of it. pyCLAD's runners split validation data off each concept's training data
(`StandardRunner(validation_fraction=...)`); the test concepts are only for the final evaluation. If you did choose
settings by looking at test scores, say so next to the results.

## Report what shaped the result

The JSON file holds everything pyCLAD components report about themselves: the seed, the dataset and its size,
the model's hyperparameters and the strategy's settings. Components you write should report theirs through
`additional_info()` too (see [Implementing your own model](custom_model.md)).

In the paper, give the settings you held fixed (see above), the seeds with mean ± spread over them, buffer sizes,
model sizes and any other budget, and the pyCLAD version (`pip show pyclad`). If you changed pyCLAD itself, say so,
since the results then depend on your version.
