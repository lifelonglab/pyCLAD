# Scenarios

### Overview

A **scenario** feeds the train concepts of a dataset to a strategy and evaluates it after each one. Scenarios
differ in what they reveal about the stream [1]:

- **Concept-aware**: concept boundaries and concept identities are known, at training and at prediction time.
- **Concept-incremental**: boundaries are known (each training step is one concept), identities are not.
- **Concept-agnostic**: neither is known; data arrives in batches and changes are unannounced.

They are adapted from the scenarios of continual learning for image classification. What a concept is, and why a
strategy may only infer what the scenario hides, is explained in [Core concepts](learn/concepts.md#scenarios);
[Choosing a scenario and a strategy](guides/choosing.md) helps match a scenario to an application.

### Scenarios and strategy contracts

Each scenario drives one strategy contract, defined in `pyclad.strategies.strategy`. A strategy can only run in the
scenarios whose contract it implements:

| Scenario | Strategy contract | What the scenario calls |
|---|---|---|
| `ConceptAwareScenario` | `ConceptAwareStrategy` | `learn(data, concept_id)` once per concept; `predict(data, concept_id)` |
| `ConceptIncrementalScenario` | `ConceptIncrementalStrategy` | `learn(data)` once per concept; `predict(data)` |
| `ConceptAgnosticScenario` | `ConceptAgnosticStrategy` | `learn(data)` on consecutive batches of 256 samples (`run(batch_size=...)`); `predict(data)` |
| `SupervisedConceptIncrementalScenario` | `SupervisedStrategy` | `learn(concept)` once per concept, so labels and masks reach the model; `predict(data)` |

In every scenario, after each training step the strategy is evaluated on all test concepts, and `predict` returns a
`PredictionResults` (`y_pred` and `anomaly_scores`). The callbacks receive the evaluated concept, its labels and both
arrays. A scenario always calls `after_scenario` on its callbacks, even when the run fails, so callbacks that hold
resources (the energy and memory trackers) release them.

The strategies that implement each contract are listed in [Strategies](strategies.md).

### Code example

The same strategy run in the three scenarios:

```python
--8<-- "docs/snippets/scenarios/three_scenarios.py:scenarios"
```

Scenarios do not check the strategy's type at run time. Pass only a strategy that implements the scenario's
contract: a strategy that accepts extra keyword arguments could otherwise silently ignore information it was not
designed to use.

### References

[1] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.
