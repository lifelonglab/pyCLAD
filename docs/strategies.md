# Strategies

## Overview

Continual learning strategies are designed to decide *when*, *how*, and using *what* data models should be updated
throughout their lifespan. Strategies are inspired by the most diverse disciplines, including neuroscience and biology,
which reveal the natural way human beings learn and retain knowledge [1].

The general goal for the model is to be able to pick up new skills, adjust to newly presented tasks, and draw on
previously learned information to tackle both new obstacles and the recurrence of previously seen tasks.
Particular interest is devoted to simultaneous adaptation and knowledge retention and a model's ability to
simultaneously handle many tasks/concepts, avoiding forgetting.

There are multiple types of strategy that approach the problem in different ways. The three most popular types of
approaches focus on:
i) replaying previously seen data, ii) modifying models architecture, and iii) adding regularization to model update
process. You can read more about them in [1].

## Available strategies

pyCLAD provides the following strategies:

- **Naive**: Models are updated as new data becomes available (as in online anomaly detection), without any smart
  continual learning strategy to tune adaptation and knowledge retention[^3].
  [^3]: By updating the model only based on the new data, a reasonable expectation is that the model will gradually or
  catastrophically forget knowledge of previously presented data. It can be considered as a lower-bound non-continual
  baseline learning strategy

- **Replay**: It preserves selected data samples from previous concepts in a memory buffer, which is limited in size by
  a *budget*. When the model faces a new concept, the replay buffer is updated to include the data from the new concept.
  As a result, the replay buffer contains knowledge of all concepts presented so far. The replay buffer is then used
  while updating the model to mitigate forgetting.

- **Cumulative**: New data is accumulated as it comes, and the model is retrained using all available data. The
  rationale for this baseline is to simulate upper-bound performance assuming full knowledge of the data, and unlimited
  computational resources to deal with stored data (storage) and model retraining (time) [^1].
  [^1]: Cumulative can also be regarded as a variant of Replay with unlimited memory. This baseline is interesting since
  it allows us to estimate the accuracy that could be achieved at a much higher computational cost.

- **MSTE**: It creates a pool or ensemble of models, each of which is an expert for a single concept. Whenever a new
  concept is presented, a new model is trained on the new data and added to the pool[^2].
  [^2]: It simulates upper-bound model performance in a non-continual scenario. We note that this is an unrealistic
  setting since, in real-world scenarios, it requires extremely high computational resources to deal with a potentially
  infinite number of models, and the availability of concept identifiers, which are not available for
  concept-incremental and concept-agnostic scenarios.

Strategies are of different types based on the scenario type (e.g. concept-incremental, concept-aware, concept-agnostic)
since scenario types determine available information on concept identifiers and boundaries. For more details see
[Scenarios](scenarios.md).

## Strategy types

There are different types of scenarios that can be considered in continual learning. The most common right now are:
Concept-aware, Concept-incremental, and Concept-agnostic. They are different in what information is available to the
strategy during the learning process (see [Scenarios](scenarios.md)).
For example, in concept-aware scenario the strategy knows the concept identifier and concept boundaries (passed as
parameters), while in concept-agnostic scenario it does not have any information about the current concept.

To make creating new strategies easier, pyCLAD provides a base class for each of the three implemented scenario types.
The base classes are: **ConceptAwareStrategy**, **ConceptIncrementalStrategy**, and **ConceptAgnosticStrategy**.

From this viewpoint, different strategy types can be suited to different scenarios. For example, considering the
strategies pre-implemented in pyCLAD:

- **MSTE** is inherently a **ConceptAwareStrategy**, as it requires information about the concept identifier to create
  a new model for each concept.

- **Naive**, **Replay**, and **CumulativeStrategy** support all three strategy types.

## How to create your own strategy?

If you want to create a new strategy for one of preexisting scenario types, you can create a new class that inherits
from one or more of the three base classes. Let's analyze the example of the *Cumulative* strategy, that simply
accumulates all data and retrains the model on all data.

``` py 
class CumulativeStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    def __init__(self, model: Model):
        self._replay = []
        self._model = model
```

As we can see, *CumulativeStrategy* inherits from all three base classes, which means that it can be used in all three
types of scenarios. This is because the strategy does not require any additional information about the currently
processed concept to work.
In the `__init__` method, we initialize the model and the replay buffer that stores all data seen so far.

Then, we have to implement `learn` methods.

``` py
    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        self._replay.append(data)
        self._model.fit(np.concatenate(self._replay))
```

Our *learn* method gets the *data* parameter, as well as any additional parameters that are passed to the strategy.
These additional parameters is required for the strategy to be able to work in any type of scenarios that can pass
additional arguments.
In the method, we append the new data to the replay buffer and retrain the model on all data seen so far.

Our next step is implementing the `predict` method that simply returns the prediction of the model.

``` py
    def predict(self, data: np.ndarray, *args, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)
```

There are also two additional methods that can be implemented in the strategy:
- *name* - it returns the name of the strategy used in the logs and output data.
- *additional_info* - it allows the strategy to add additional information to the logs and output data.

## References

[1] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural
networks: A review. *Neural networks*, 113, 54-71.

[2] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection:
New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.

