# Metrics 

### Overview
Continual learning scenarios require a continuous evaluation across all concepts.
To this end, pyCLAD adopts an evaluation protocol that considers the performance across all concepts in each scenario: 

- Initializes a matrix $R$ to accommodate anomaly detection results for specific tasks
- Iterates over training sets for all concepts 
- For each concept, trains/updates the model and evaluates it on all testing sets for all concepts, i.e. previous, current, and future concepts.
- Yields the resulting matrix  $R$, where entries $R_{i, j}$ define the performance of the model evaluated on concept $j$ after learning concept $i$. 
The matrix $R$ can be used to directly compute continual learning metrics.

### Supported metrics
pyCLAD supports three main metrics:

- **Continual Average** (CA): It assesses models' performance on all concepts after learning every new concept, instead of models' performance on just a single concept. 
It is general, since it operates on the matrix $R$ and it can support any target metric of choice, such as F1-Score and ROC-AUC[^1]. It is defined as: 

$\text{CA} = \frac{\sum_{i \ge j}^N R_{{i,j}}}{\frac{N(N+1)}{2}}$
[^1]: Note: ROC-AUC is sometimes preferred over threshold-dependent metrics such as Precision, Recall, and F--Score, since it allows us to evaluate the model's performance more comprehensively. ROC--AUC may be swapped with other metrics of choice without impacting the validity of the protocol. 

- **Backward Transfer** (BWT): Measures the impact of learning new concepts on the performance of all previously learned concepts[^2]. It is computed as:

$\text{BWT} = \frac{\sum_{i=2}^N\sum_{j=1}^{i-1} R_{i, j} - R_{j,j}}{\frac{N(N-1)}{2}}$

[^2]: Negative backward transfer suggests that the model is prone to forgetting. A strongly negative value is also sometimes regarded as catastrophic forgetting. On the other hand, positive backward transfer suggests that learning new concepts benefits models' performance on previously learned concepts.  

- **Forward Transfer** (FWT): Measures the impact of learning each concept on the model's performance on future concepts[^3].  It is computed as:

$\text{FWT} = \frac{\sum_{i<j}^{N} R_{i, j}}{\frac{N(N-1)}{2}}$

[^3]: It can also be thought of as the zero-shot model performance on future concepts since it assesses model performance on unseen concepts. It partially depends on concept similarity (task similarity) and the model's knowledge transfer ability.




### Rectangular matrices and step schedules

When training steps group several concepts (see [Step Schedule](datasets.md)), the rows of $R$ are
training steps and the columns are evaluated concepts, so $R$ becomes rectangular ($T \times N$
with $T < N$) and the two axes no longer share names.

The three metrics above walk the diagonal or the triangles of $R$, which is only meaningful when
each training step corresponds to exactly one evaluated concept. Handed a rectangular matrix they
raise `ValueError` instead of returning a number computed over cells that do not mean what the
formula assumes.

For rectangular matrices pyCLAD provides metrics that take one extra argument: $s_k$, the index of
the training step at which evaluated concept $k$ first entered training. Rows above $s_k$ describe
the model *before* it ever saw that concept, which is a different quantity from forgetting. Indices
below are 0-based, so the final training step is $T-1$.

- **Final Step Average** (FSA): the average across all evaluated concepts after the last training
step. This is the `A-AUROC` figure reported by CDAD-style papers. It reads only the last row, so it
works on square and rectangular matrices alike:

$\text{FSA} = \frac{1}{N}\sum_{k=0}^{N-1} R_{T-1, k}$

- **Schedule-Aware Forgetting Measure**: forgetting restricted to the rows in which a concept had
already been trained. Concepts with $s_k \ge T-1$ are skipped, since a concept that enters training
only at the final step cannot have been forgotten yet:

$f_k = \max_{j \in [s_k,\, T-2]} R_{j, k} - R_{T-1, k}$

- **Schedule-Aware Forward Transfer**: the model's performance on concepts it has not been trained
on yet, averaged over the pre-training rows. Concepts with $s_k = 0$ are skipped, and so are
individual cells where the base metric was undefined, so the mean is taken over the non-NaN
pre-training rows rather than over all $s_k$ of them:

$\text{fwt}_k = \underset{j \,\in\, [0,\, s_k - 1]}{\text{mean}} R_{j, k}$

- **Schedule-Aware New Task Acquisition**: performance on a concept right after it first entered
training, before any later step could interfere. Together with the forgetting measure it separates
a plasticity problem (the concept was never learned) from a stability one (it was learned, then
lost). On a square matrix with one concept per step this reads the diagonal:

$\text{nta}_k = R_{s_k, k}$

Wiring them up is one argument, because the grouped dataset carries the mapping:

```python
callback = ConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[FinalStepAverage()],
    schedule_aware_metrics=[
        ScheduleAwareForgettingMeasure(),
        ScheduleAwareForwardTransfer(),
        ScheduleAwareNewTaskAcquisition(),
    ],
    first_seen_step=scheduled_dataset.first_seen_step(),
)
```

The evaluation protocol slightly differs based on the scenario. Specifically:

- **Concept-aware** and **concept-incremental**: batches $T_i$ (training) and $E_i$ (evaluation) correspond to the single $i-$th concept.

- **Concept-agnostic**: a batch does not necessarily correspond to a single concept since the setting assumes that no explicit concept boundaries are provided to the lifelong algorithm.
As a result, the evaluation may require considering multiple batches as belonging to the same concept or a single batch including data for more than one concept.
