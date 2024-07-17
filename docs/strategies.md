# Strategies

### Overview 

Continual anomaly detection strategies focus on specific objectives which can be compared to other types of approach. One possible categorization of anomaly detection approaches is the following:

- **Offline**: Models are trained once on background data and do not require updates (examples: post-incident analysis, breast cancer detection)  and  on updating the model as new data is observed, assuming only the most recent information is relevant. This approach is static in nature and does not provide adaptation. 

- **Online**: Models are updated as new data is observed, assuming that the most recent information is the most relevant. This approach is popular in real-world dynamic applications where adaptation is necessary, but makes models prone to forgetting past knowledge. 

- **Continual**: Models are updated to simultaneously consider *adaptation* to new conditions and *knowledge retention* of previously observed (and potentially recurring) conditions [1].
This behavior attempts to overcome limitations of both offline and online anomaly detection in complex scenarios.

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


### References

[1] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural networks*, 113, 54-71.

[2] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.

[3] Buzzega, P., Boschini, M., Porrello, A., & Calderara, S. (2021, January). Rethinking experience replay: a bag of tricks for continual learning. In *2020 25th International Conference on Pattern Recognition (ICPR)* (pp. 2180-2187). IEEE.