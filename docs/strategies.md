# Strategies

### Overview 

Continual learning strategies are designed to decide *when*, *how*, and using *what* data models should be updated throughout their lifespan. Strategies are inspired by the most diverse disciplines, including neuroscience and biology, which reveal the natural way human beings learn and retain knowledge [1].

The general goal for the model is to be able to pick up new skills, adjust to newly presented tasks, and draw on previously learned information to tackle both new obstacles and the recurrence of previously seen tasks.
Particular interest is devoted to simultaneous adaptation and knowledge retention and a model's ability to simultaneously handle many tasks/concepts, avoiding forgetting.

### Available strategies
The following strategies are implemented in pyCLAD:

- **Naive**: Models are updated as new data becomes available (as in online anomaly detection), without any smart lifelong learning strategy to tune adaptation and knowledge retention[^3].
[^3]: By updating the model only based on the new data, a reasonable expectation is that the model will gradually or catastrophically forget knowledge of previously presented data. It can be considered as a lower-bound non-continual baseline learning strategy

- **Replay**: It preserves selected data samples from previous concepts in a memory buffer, which is limited in size by a *budget*. When the model faces a new concept, the replay buffer is updated to include the data from the new concept. As a result, the replay buffer contains knowledge of all concepts presented so far. The replay buffer is then used while updating the model to mitigate forgetting [3]. 

- **Cumulative**: New data is accumulated as it comes, and the model is retrained using all available data. The rationale for this baseline is to simulate upper-bound performance assuming full knowledge of the data, and unlimited computational resources to deal with stored data (storage) and model retraining (time) [^1]. 
[^1]: Cumulative can also be regarded as a variant of Replay with unlimited memory. This baseline is interesting since it allows us to estimate the accuracy that could be achieved at a much higher computational cost.

- **MSTE**: A pool or ensemble of models, each of which is an expert for a single concept, is adopted. Whenever a new concept is presented, a new model is trained on the new data and added to the pool[^2].
[^2]: It simulates upper-bound model performance in a non-continual scenario. We note that this is an unrealistic setting since, in real-world scenarios, it requires extremely high computational resources to deal with a potentially infinite number of models, and the availability of concept identifiers, which are not available for concept-incremental and concept-agnostic scenarios.

Strategies are of different types based on the scenario type (e.g. concept-incremental, concept-aware, concept-agnostic) since scenario types determine available information on concept identifiers and boundaries. For more details see Scenarios.

### Strategy types
Strategies can be implemented referring to the three base classes pre-implemented in pyCLAD: **ConceptAwareStrategy**, **ConceptIncrementalStrategy**, and **ConceptAgnosticStrategy**.

Different strategy types can exploit different data assumptions and availability to maximize their effectiveness in different scenarios (see [Scenarios](scenarios.md)).  
One of the key differences is that a **ConceptAwareStrategy** strategy can leverage information about the concept identifier during the prediction stage, which is not available in the other two strategy types.

From this viewpoint, different strategy types can be suited to different scenarios. For example, considering the strategies pre-implemented in pyCLAD:  

- **MSTE** is inherently a **ConceptAwareStrategy**

- **Naive**, **Replay**, and **CumulativeStrategy** support all three strategy types

### References

[1] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural networks*, 113, 54-71.

[2] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.

[3] Buzzega, P., Boschini, M., Porrello, A., & Calderara, S. (2021, January). Rethinking experience replay: a bag of tricks for continual learning. In *2020 25th International Conference on Pattern Recognition (ICPR)* (pp. 2180-2187). IEEE.