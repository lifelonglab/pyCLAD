# Implementing your own strategy

A strategy decides how a model is updated as concepts arrive. It implements one or more of the contracts in
`pyclad.strategies.strategy`, one per scenario it can run in (see [Strategies](../strategies.md)).

## Choose the contracts honestly

Implement a contract only if the method can work with what that scenario gives it:

- `ConceptAwareStrategy`: `learn(data, concept_id)` and `predict(data, concept_id)`. Only if the method uses the
  concept identity.
- `ConceptIncrementalStrategy`: `learn(data)`, where each call is one whole concept.
- `ConceptAgnosticStrategy`: `learn(data)` on arbitrary batches, with no boundaries.

A method may infer what it is not given from the data itself, for example by detecting drift. It may not get it any
other way: not from the dataset object, concept names, labels or the order of evaluation.

## Write the strategy

This strategy keeps the most recent samples and retrains on them. It needs neither boundaries nor identities, so it
implements the incremental and agnostic contracts:

```python
--8<-- "docs/snippets/guides/custom_strategy.py:strategy"
```

Decide explicitly what carries over between `learn()` calls, and say it in the code: here the window deliberately
spans concepts.

## Run it

```python
--8<-- "docs/snippets/guides/custom_strategy.py:run"
```

## Test it

pyCLAD's smoke tests run a strategy through a three-concept scenario with a small autoencoder and check that every
concept was evaluated and every score is defined. Subclass `BaseStrategyTest` and provide the strategy:

```python
--8<-- "docs/snippets/guides/custom_strategy_tests.py:test"
```

Add tests of the behaviour that makes your method what it is: for a buffer, that it holds what it should after
several concepts; for a regularizer, that the penalty is zero before the second concept.

## Checklist

- [ ] Only the contracts the method can honestly satisfy are implemented.
- [ ] `learn` sees only the training data it is given. `predict` does not change learned state.
- [ ] State kept between `learn()` calls is deliberate, and it is clear whether and when it resets.
- [ ] `additional_info()` reports every setting that shapes the result: buffer size, regularization weight, the
      model's name.
- [ ] Randomness comes from a `seed` or `rng` argument, defaulting to `pyclad.seed.new_generator()`.
- [ ] A port of a published method cites the paper in the class docstring and lists every departure from it in the
      docs.
- [ ] To contribute it: a smoke test in `tests/strategies/smoke_tests/`, an example in `examples/strategies/`, and an
      entry in [Strategies](../strategies.md).
