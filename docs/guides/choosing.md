# Choosing a scenario and a strategy

## Start from what your application knows

The scenario should reflect what a deployed detector would know about the stream. Pick it from your application, not
from which scenario gives the best numbers:

| In your application… | Scenario |
|---|---|
| you know which concept each sample belongs to, both during training and when detecting (for example, which machine or which customer the data comes from) | `ConceptAwareScenario` |
| you know *when* the concept changes, but not which concept it is (for example, a new production run starts, but nobody labels which profile it uses) | `ConceptIncrementalScenario` |
| you know neither: data arrives as a stream and changes are unannounced | `ConceptAgnosticScenario` |
| as concept-incremental, but your detector trains on labelled anomalies too | `SupervisedConceptIncrementalScenario` |

A harder scenario is not a worse one: it is a different question. Results from different scenarios answer different
questions and should not share a table without saying so. See [Scenarios](../scenarios.md).

## Pick strategies that fit the scenario

Only strategies that implement the scenario's contract can run in it. The table in "Available strategies" in
[Strategies](../strategies.md#available-strategies) shows which scenarios each strategy supports and which models it
needs: EWC, LwF, DER++ and A-GEM need a PyTorch model, while the others work with any model, including PyOD
detectors.

## Always include the baselines

A continual strategy is only interesting relative to the trivial ones:

- **Naive** (retrain on the new data only) shows how much the model forgets on its own: the lower bound.
- **Cumulative** (retrain on all data so far) shows what is achievable without memory or compute limits: an upper
  bound for strategies that keep a bounded memory.
- **MSTE** (one model per concept) is the upper bound when concept identities are known, so include it in the
  concept-aware scenario.

If a strategy does not beat Naive, it does not mitigate forgetting on that benchmark. If it matches Cumulative with
a fraction of the memory, that is the result worth reporting. Report memory or buffer sizes next to the scores.

## Check that the benchmark can show forgetting

Forgetting only shows up when concepts differ enough for a model trained on one to fail on another. Look at the
metric matrix of the Naive baseline: if the off-diagonal cells below the diagonal are as good as the diagonal, the
benchmark cannot separate strategies by how much they forget, whatever their scores.

## Next

- [Running a reproducible comparison](experiments.md): seeds, settings and reporting.
- [Using built-in datasets and step schedules](datasets.md).
