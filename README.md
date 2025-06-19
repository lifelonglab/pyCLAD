# pyCLAD - Python Continual Lifelong Anomaly Detection

## What is pyCLAD?

**pyCLAD is a unified framework for continual anomaly detection.**
Its main goal is to foster successful scientific development in continual anomaly detection by
providing robust implementations of strategies, models, scenarios, and metrics, complemented with code examples and user
documentation.
pyCLAD also facilitates the design and implementation of experimental pipelines, providing a streamlined, unified, and
fully reproducible execution workflow.
pyCLAD is built following a component-oriented design approach. As a result, it provides an extensible approach, where
users are not required to reimplement foundational aspects of the code, and can just focus on extending specific
classes.

## How to use pyCLAD?

### Installation

pyCLAD is provided as a Pyton package available in `pypi`. Therefore, you can install it as a package using tools such
as pip and conda, for example:

`pip install pyclad`.

Moreover, the source code is available in [the GitHub repository](https://github.com/lifelonglab/pyCLAD).

#### Optional dependencies

Depending on the anomaly detection models you want to use, you may need to install additional packages,
such as `tensorflow` and `pytorch`.
We do not include them in default installation to avoid putting heavy dependencies for the core installation.
pyCLAD supports the use of any model from pyOD library, some of which may require installation of additional packages (
see [pyOD docs](https://pyod.readthedocs.io/en/latest/).

### Getting started

There are a few valuable resources supporting getting started with pyCLAD:

- [Getting started guide](https://pyclad.readthedocs.io/en/latest/getting_started/)
- [Documentation](https://pyclad.readthedocs.io/en/latest/)
- [Examples available in github repository](https://github.com/lifelonglab/pyCLAD/tree/main/examples)

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

### Quick example

```python
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
callbacks = [
    ConceptMetricCallback(
        base_metric=RocAuc(),
        metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
    ),
    TimeEvaluationCallback(),
]

# Execute the concept agnostic scenario
scenario = ConceptAgnosticScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
scenario.run()

# Save the results
output_writer = JsonOutputWriter(pathlib.Path("output.json"))
output_writer.write([model, dataset, strategy, *callbacks])
```

**[See more examples here](https://github.com/lifelonglab/pyCLAD/tree/main/examples)**

## Citing pyCLAD

A paper describing pyCLAD is published in SoftwareX Journal - [click here](https://www.sciencedirect.com/science/article/pii/S2352711024003649). If you use pyCLAD in your research, please cite:

```bibtex
@article{faber2025pyclad,
  title = {pyCLAD: The universal framework for continual lifelong anomaly detection},
  journal = {SoftwareX},
  volume = {29},
  pages = {101994},
  year = {2025},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2024.101994},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711024003649},
  author = {Kamil Faber and Bartlomiej Sniezynski and Nathalie Japkowicz and Roberto Corizzo},
  keywords = {Continual anomaly detection, Lifelong anomaly detection, Continual learning, Anomaly detection, Software},
}
```

## How to contribute?

We welcome all contributions! If you want to contribute to pyCLAD, please follow the guidelines in
the [CONTRIBUTING.md](https://github.com/lifelonglab/pyCLAD/tree/main/CONTRIBUTING.md) file.