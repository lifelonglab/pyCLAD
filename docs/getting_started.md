# Getting Started

In this page, we describe the core concepts of pyCLAD and provide a quick example with line by line explanation to help
you get started.

## Overview

pyCLAD is built upon a few core concepts:

- **Scenario**: a continual scenario defines the data stream so that it reflects different real-life conditions and what
  are
  the challenges faced by continual strategy.
- **Strategy**: a strategy is a way to manage model updates. Continual strategy is responsible for how, when, and with
  which
  data models should be updated. Its aim is to introduce knowledge retention while keeping the ability to adapt.
- **Model**: a model is a machine learning model used for anomaly detection. Models are often leveraged by continual
  strategies that add additional layer of managing model's updates.
- **Dataset**: a dataset is a collection of data used for training and evaluation of the model.
- **Metrics**: a metric is a way to evaluate the performance of the model.
- **Callbacks**: a callback is a function that is called at specific points during the scenario. Callbacks are
  useful for monitoring the process, calculating metrics, and more.

If you want to learn more about each of these components, please refer to the specific pages in the documentation.

## How to run your first experiment?

To get started, we describe a typical pyCLAD execution step-by-step.
In the example below, we run an experiment on randomly generated data in a *Concept-agnostic scenario*, using a
*Cumulative strategy* and a *One-Class SVM model*.

### Preparing a dataset

pyCLAD leverages the idea of concepts to simulate continual learning scenarios. A concept is a subset of data that
represents a specific distribution/activity/task.
See more in [Datasets](datasets.md).
Every dataset is represented as a sequence of concepts, where each concept is represented by a *name*, *data*, and (
optionally) *labels*.
In this example, we use randomly generated data to demonstrate the workflow and manual concept creation. However, we
also provide examples with real-world datasets (see more
info [here](getting_started.md#leveraging-real-world-datasets)).

Let's start our example with creating a few concepts with randomly generated data:

``` py
concept1_train = Concept("concept1", data=np.random.rand(100, 10))
concept1_test = Concept("concept1", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

concept2_train = Concept("concept2", data=np.random.rand(100, 10))
concept2_test = Concept("concept2", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

concept3_train = Concept("concept3", data=np.random.rand(100, 10))
concept3_test = Concept("concept3", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))
```

Having created the concepts, we can build a dataset based on them:

``` py
dataset = ConceptsDataset(
    name="GeneratedDataset",
    train_concepts=[concept1_train, concept2_train, concept3_train],
    test_concepts=[concept1_test, concept2_test, concept3_test],
)
```

You can leverage any of your preferred datasets by dividing it into concepts. We provide an example of how to extract
concepts from a dataset in the [Datasets](datasets.md) section.

### Defining a strategy

A continual learning strategy manages model updates to ensure knowledge retention while adapting to new data.
pyCLAD provides a few baseline strategies, such as *Cumulative*, *MSTE*, and *Naive*, as well as very commonly adopted
*Replay* (see more info about strategies in [Strategies](strategies.md)).

As strategies manage model updates, they require a model as an input. In this example, we use an *One Class SVM*
model (see more info about models in [Models](models.md)).

Let's start with defining a model:

``` py
model = OneClassSVMAdapter()
```

Then, we can create a strategy:

``` py
strategy = CumulativeStrategy(model)
```

### Defining callbacks

Callbacks allow to effectively log useful information in pyCLAD during key checkpoints (*before* and *after*) of the
experimental workflow (see more in [Callbacks](callbacks.md)).
In this example, we leverage the callbacks that monitor model performance (*ConceptMetricCallback*) and
execution time (*TimeEvaluationCallback*).

Let's start with defining time evaluation callback:

``` py
time_callback = TimeEvaluationCallback()
```

Then, let's define a callback that logs model performance:

``` py
metric_callback = ConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
)
```

*ConceptMetricCallback* takes as an input:

- *base_metric*: a non-continual base metric that is used to evaluate the performance of single concept, for example
  ROC-AUC.
- *summarized_metrics*: a list of continual learning metrics that summarize the whole scenario as a single value,
  calculated with *base_metric* as a base.
- *stepwise_metrics* (optional): a list of continual learning metrics that report one value after each learned
  concept, such as `ForgettingMeasure`.

See more in [Metrics](metrics.md).

### Running a scenario

Finally, having defined the dataset, strategy, and callbacks, we can run a scenario.
A continual learning **scenario** defines data assumptions that have implications on how *strategies* and *models*
behave.
In this example, we run a *Concept-agnostic scenario* (see more in [Scenarios](scenarios.md)).
To do this, we start with creating a scenario passing the dataset, strategy, and callbacks as arguments. Then, we use
the `run` method to execute the scenario.

``` py
scenario = ConceptAgnosticScenario(dataset=dataset, strategy=strategy,
                                   callbacks=[metric_callback, time_callback])
scenario.run()
```

### Saving results

Last but not least, we need to save the results of the experiment gathered by the callbacks, along with parameters of
dataset, strategy, and model.
We can use the *JsonOutputWriter* to save the results in JSON format.

``` py
output_writer = JsonOutputWriter(pathlib.Path("output.json"))
output_writer.write([model, dataset, strategy, metric_callback, time_callback])
```

An example of output in JSON format (*output.json*) is partially shown below. We can see info about the dataset,
strategy, model, and metrics calculated by the callbacks.

```json
{
  "model": {
    "name": "OneClassSVM",
    "kernel": "rbf",
    ...
  },
  "dataset": {
    "name": "GeneratedDataset",
    "train_concepts_no": 3,
    "test_concepts_no": 3
  },
  "strategy": {
    "name": "Cumulative",
    "model": "OneClassSVM",
    "buffer_size": 300
  },
  "concept_metric_callback_ROC-AUC": {
    "base_metric_name": "ROC-AUC",
    "metrics": {
      "ContinualAverage": 0.5016447508645475,
      "BackwardTransfer": -0.006135787648392695,
      "ForwardTransfer": 0.5108682749081299
    },
    ...
  },
  "time_evaluation_callback": {
    "time_by_concept": {
      "concept1": {
        "train_time": 0.013182401657104492,
        "eval_time": 0.00678563117980957
      },
      ...
    },
    "train_time_total": 0.021419048309326172,
    "eval_time_total": 0.02041482925415039
  }
}
```

### Making a run repeatable

Strategies and models draw random numbers from Python's `random` module, from NumPy and from PyTorch. Call
`set_seed` once at the start of a script to seed all three:

``` py
from pyclad.seed import set_seed

seed = set_seed(42)
```

Components that take their own `seed` or `rng` argument keep using it; the ones given none follow this seed.
`set_seed` returns an object you can pass to the output writer, so the seed is saved with the results:

``` py
output_writer.write([seed, model, dataset, strategy, metric_callback, time_callback])
```

## Full code example

The complete script, including the imports, is shown below. You can see this and more code examples in the
[repository](https://github.com/lifelonglab/pyCLAD/tree/main/examples).

```python linenums="1"
import pathlib

import numpy as np

from pyclad.callbacks import ConceptMetricCallback, TimeEvaluationCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import BackwardTransfer, ContinualAverage, ForwardTransfer, RocAuc
from pyclad.models.adapters.pyod_adapters import OneClassSVMAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios import ConceptAgnosticScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

# Prepare random data for 3 concepts
concept1_train = Concept("concept1", data=np.random.rand(100, 10))
concept1_test = Concept("concept1", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

concept2_train = Concept("concept2", data=np.random.rand(100, 10))
concept2_test = Concept("concept2", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

concept3_train = Concept("concept3", data=np.random.rand(100, 10))
concept3_test = Concept("concept3", data=np.random.rand(100, 10), labels=np.random.randint(0, 2, 100))

# Build a dataset based on the previously created concepts
dataset = ConceptsDataset(
    name="GeneratedDataset",
    train_concepts=[concept1_train, concept2_train, concept3_train],
    test_concepts=[concept1_test, concept2_test, concept3_test],
)
# Define model, strategy, and callbacks
model = OneClassSVMAdapter()
strategy = CumulativeStrategy(model)

time_callback = TimeEvaluationCallback()
metric_callback = ConceptMetricCallback(
    base_metric=RocAuc(),
    summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
)

# Execute the concept agnostic scenario
scenario = ConceptAgnosticScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback, time_callback])
scenario.run()

# Save the results
output_writer = JsonOutputWriter(pathlib.Path("output.json"))
output_writer.write([model, dataset, strategy, metric_callback, time_callback])
```

## Leveraging real-world datasets

In the example above, we used randomly generated data to demonstrate the workflow. However, in [Examples](examples.md),
we also showcase examples with real-world, such as
datasets ([UNSW](https://github.com/lifelonglab/pyCLAD/blob/main/examples/unsw_dataset_example.py)
and [Energy](https://github.com/lifelonglab/pyCLAD/blob/main/examples/energy_dataset_example.py)).
pyCLAD provides a few datasets that can be used out-of-the-box, such as *UNSW-NB15*, *NSL-KDD*, *Wind Energy*, and
*Energy Plants*. They are available as classes (for example `UnswDataset`) and automatically downloaded from hugging
face. See more info in [Datasets](datasets.md).

Moreover, pyCLAD provides out-of-the-box loader (`read_dataset_from_npy`) for continual learning scenarios extracted
leveraging the algorithm described [in this paper](https://ieeexplore.ieee.org/abstract/document/10473036) (
see [code](https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios)). 