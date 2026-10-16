# Strategies

### Overview

Continual learning strategies decide *when*, *how* and *with what data* a model is updated throughout its lifespan.
Their goal is to let the model pick up new concepts while retaining what it learned about earlier ones, including
concepts that recur later [1, 2].

A strategy implements one or more of the contracts in `pyclad.strategies.strategy`, one for each scenario it can run in
(see [Scenarios](scenarios.md)):

- **ConceptAwareStrategy**: `learn(data, concept_id)` and `predict(data, concept_id)`. The strategy knows which
  concept it is learning and which one it is evaluated on.
- **ConceptIncrementalStrategy**: `learn(data)` and `predict(data)`. Each `learn` call is one whole concept, but the
  concept is not named.
- **ConceptAgnosticStrategy**: `learn(data)` and `predict(data)`. Data arrives in batches with no boundaries.
- **SupervisedStrategy**: `learn(concept)` receives the whole `Concept`, so that labels (and, for images, masks)
  reach the model.

A strategy implements only the contracts it can honestly satisfy. For example, MSTE needs the concept identifier to
pick a model, so it is concept-aware only.

### Available strategies

| Strategy | Class | Concept-aware | Concept-incremental | Concept-agnostic | Model |
|---|---|:-:|:-:|:-:|---|
| Naive | `NaiveStrategy` | ✓ | ✓ | ✓ | any `Model` |
| Cumulative | `CumulativeStrategy` | ✓ | ✓ | ✓ | any `Model` |
| MSTE | `MSTE` | ✓ | | | any `Model` |
| Replay (buffer only) | `ReplayOnlyStrategy` | ✓ | ✓ | | any `Model` |
| Replay (buffer + new data) | `ReplayEnhancedStrategy` | ✓ | ✓ | ✓ | any `Model` |
| VLAD | `VladConceptAgnosticStrategy`, `VladConceptIncrementalStrategy` | | ✓ | ✓ | any `Model` |
| EWC | `EWCStrategy` | ✓ | ✓ | | `TorchBackbone` |
| LwF | `LwFStrategy` | | ✓ | ✓ | `TorchBackbone` |
| DER++ | `DerPlusPlus` | | | ✓ | `TorchBackbone` |
| A-GEM | `AGEMStrategy` | ✓ | ✓ | ✓ | `TorchBackbone` |

The strategies in the first group work with any model, because they only decide which data the model is trained
on. The regularization and gradient-projection strategies change the training loss or the gradients, so they need a
PyTorch model (`TorchBackbone`) and a runner that owns the training loop; see [Models](models.md). They need the
`torch` extra.

Image-specific strategies (`UCADStrategy`, `ReplayCADStrategy`, `NaiveSupervisedStrategy`) are described in
[Vision](vision.md).

### Baselines

- **Naive** retrains the model on each new batch of data only, as in online anomaly detection. It is expected to
  forget earlier concepts, and serves as a lower bound.
- **Cumulative** keeps all data seen so far and retrains the model on all of it. It approximates the upper bound
  achievable with unlimited memory and compute. In the concept-agnostic scenario it retrains on every batch, which gets
  slow on large datasets.
- **MSTE** (Multiple Single-Task Experts) trains a separate model for each concept and, at prediction time, uses the
  model of the concept being evaluated. It is an upper bound for the concept-aware scenario, where concept identifiers
  are available at both training and prediction time. For a concept it has no model for yet, it returns all-zero
  scores, so its results on future concepts (and therefore its forward transfer) are those of a constant score: 0.5
  ROC-AUC.

### Replay

Replay strategies keep a bounded memory buffer of past samples and train on it to mitigate forgetting [3].

- **ReplayOnlyStrategy** adds the new data to the buffer first, then trains the model on the buffer only.
- **ReplayEnhancedStrategy** trains the model on the buffer plus all of the new data, then adds the new data to the
  buffer.

Both take a `ReplayBuffer`. `AdaptiveBalancedReplayBuffer(selection_method, max_size)` gives each `update` call (each
concept, or each batch in the concept-agnostic scenario) an equal share of `max_size`, and shrinks the older shares
as new ones arrive. The samples kept are chosen by the `selection_method`, for example `RandomSelection`.

```python
--8<-- "docs/snippets/strategies/model_agnostic.py:strategies"
```

### Regularization and gradient projection

These strategies need a `TorchBackbone`, such as `Autoencoder` or `VariationalAutoencoder`, and a `TorchRunner`,
such as `StandardRunner`:

- **EWC** (Elastic Weight Consolidation) [4] estimates how important each parameter was for earlier concepts (the
  diagonal Fisher information) and penalizes changing important parameters. `lambda_ewc` sets the strength of the
  penalty; `online=True` keeps a single running estimate instead of one per concept.
- **LwF** (Learning without Forgetting) [5] keeps a copy of the previous model and adds a distillation loss that keeps
  the new model's outputs close to the old ones. `alpha` weighs the distillation term.
- **DER++** (Dark Experience Replay++) [6] keeps a reservoir buffer of past inputs with the model's outputs on them,
  and adds two terms to the loss: `alpha` keeps the outputs on past inputs close to the stored ones, and `beta` replays
  the past inputs with the ordinary loss.
- **A-GEM** (Averaged Gradient Episodic Memory) [7] computes a reference gradient on samples from a reservoir buffer
  and, when the gradient on the new data conflicts with it, projects the new gradient so that the loss on the buffer
  does not increase.

DER++ and A-GEM use `ReservoirBuffer(max_capacity=...)`, which keeps a uniform sample of everything seen.

```python
--8<-- "docs/snippets/strategies/torch_strategies.py:backbone"
```

```python
--8<-- "docs/snippets/strategies/torch_strategies.py:strategies"
```

### VLAD

**VLAD** [8] detects concept changes on its own, classifies each concept as new or recurring against a hierarchical
memory, and retrains the model on the memory's replay data plus the new data. It works with any model. There is one
class per scenario, each building the change point detector it needs:

- `VladConceptAgnosticStrategy` detects changes online with a Wasserstein-distance change point detector. This is VLAD
  as published.
- `VladConceptIncrementalStrategy` trusts the scenario's concept boundaries and only decides whether each concept is
  new or recurring.

Build the memory with the matching factory, `VladMemory.for_concept_agnostic(...)` or
`VladMemory.for_concept_incremental(...)`, which picks the distance validated for that mode:

```python
--8<-- "docs/snippets/strategies/vlad.py:vlad"
```

### References

[1] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54-71.

[2] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.

[3] Buzzega, P., Boschini, M., Porrello, A., & Calderara, S. (2021). Rethinking experience replay: a bag of tricks for continual learning. In *2020 25th International Conference on Pattern Recognition (ICPR)* (pp. 2180-2187). IEEE.

[4] Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

[5] Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12), 2935-2947.

[6] Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark experience for general continual learning: a strong, simple baseline. In *Advances in Neural Information Processing Systems*, 33.

[7] Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient lifelong learning with A-GEM. In *International Conference on Learning Representations (ICLR)*.

[8] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2023). VLAD: Task-agnostic VAE-based lifelong anomaly detection. *Neural Networks*, 165, 248-273.
