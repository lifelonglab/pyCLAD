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




The evaluation protocol slightly differs based on the scenario. Specifically:

- **Concept-aware** and **concept-incremental**: batches $T_i$ (training) and $E_i$ (evaluation) correspond to the single $i-$th concept.

- **Concept-agnostic**: a batch does not necessarily correspond to a single concept since the setting assumes that no explicit concept boundaries are provided to the lifelong algorithm.
As a result, the evaluation may require considering multiple batches as belonging to the same concept or a single batch including data for more than one concept.
