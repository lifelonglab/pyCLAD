# pyCLAD

## What is pyCLAD?

pyCLAD is a unified framework for continual anomaly detection. It provides tested implementations of the parts every
experiment needs (scenarios, strategies, models, benchmarks and metrics), so that researchers can focus on their own
method instead of re-implementing, and possibly getting wrong, the evaluation around it. Every run is saved with
everything that shaped its result, which makes it repeatable and comparable.

Adding a model, strategy, dataset or metric means subclassing one class. The interfaces follow scikit-learn
(`fit` / `predict`) and work with PyOD detectors and PyTorch networks.

## What is continual anomaly detection?

**Anomaly detection** finds deviations from the normal behaviour of a process, for example in cyber-physical
systems, health monitoring or network traffic. **Continual learning** studies models that keep adapting to new
conditions while retaining what they learned before. Continual anomaly detection combines the two. Compared with
other ways of keeping a detector up to date:

- **Offline**: the model is trained once and never updated. It cannot adapt when normal behaviour changes.
- **Online**: the model is updated as new data arrives, assuming the most recent data is the most relevant. It
  adapts, but forgets earlier conditions.
- **Continual**: the model *adapts* to new conditions and *retains* knowledge of earlier ones, which matters when
  conditions recur.

To learn more, we recommend [this open-access paper](https://ieeexplore.ieee.org/abstract/document/10473036/).

## Where do I start?

1. [Install pyCLAD](learn/installation.md), with the extras you need.
2. Run [your first experiment](getting_started.md): a complete script, explained step by step.
3. Read the [core concepts](learn/concepts.md): what a concept, scenario, strategy and model are, and how pyCLAD
   evaluates them.
4. Browse the [examples](examples.md), then the reference pages for the components you use.

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
