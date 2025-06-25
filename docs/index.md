### What is pyCLAD?

pyCLAD is a unified framework for continual anomaly detection. Its main goal is to foster successful scientific
development in continual anomaly detection by providing robust implementations of common functionalities for continual
anomaly detection off-the-shelf,  minimizing the risk of error-prone tasks and fostering replicability. pyCLAD also
facilitates the design and implementation of experimental pipelines, providing a streamlined,
unified, and fully reproducible execution workflow.

It also provides a simple and convenient infrastructure for designing new strategies, models, and evaluation procedures,
enabling researchers to avoid repetitive tasks and allowing them to focus on creative and scientific aspects, reducing
the friction related to low-level implementation aspects.

The core coding infrastructure is influenced by PyTorch, a very popular framework for deep learning as well as SkLearn,
the leading machine learning and data analysis library in Python.

### What is continual anomaly detection?

**Continual Learning** puts emphasis on models that answering the need for machine learning models that continuously
adapt to new challenges in dynamic environments while retaining past knowledge.

**Anomaly Detection** is the process of detecting deviations from the normal behavior of a process, and has a very wide
range of applications including monitoring cyber-physical systems, human conditions, as well as network traffic.

**Continual anomaly detection** lies at the intersection of these two fields. Its strategies focus on specific goals
which can be compared to other types of anomaly detection approaches. One possible categorization is the following:

- **Offline**: Models are trained once on background data and do not require updates (examples: post-incident analysis,
  breast cancer detection)  and on updating the model as new data is observed, assuming only the most recent information
  is relevant. This approach is static in nature and does not provide adaptation.

- **Online**: Models are updated as new data is observed, assuming that the most recent information is the most
  relevant. This approach is popular in real-world dynamic applications where adaptation is necessary, but makes models
  prone to forgetting past knowledge.

- **Continual**: Models are updated to simultaneously consider *adaptation* to new conditions and *knowledge retention*
  of previously observed (and potentially recurring) conditions.
  This behavior attempts to overcome limitations of both offline and online anomaly detection in complex scenarios.

If you want to learn more about continual anomaly detection, we
recommend [this open-access paper](https://ieeexplore.ieee.org/abstract/document/10473036/).

### How do I install pyCLAD?

pyCLAD is available as a [Python package on PyPI](https://pypi.org/project/pyclad/). Therefore, it can be installed
using tools such as pip and conda.

#### Conda

```
conda install -c conda-forge pyclad
```

#### Pip

```
pip install pyclad
```

#### Optional dependencies

Depending on the anomaly detection models you want to use, you may need to install additional packages,
such as `tensorflow` and `pytorch`.
We do not include them in default installation to avoid putting heavy dependencies for the core installation.
pyCLAD supports the use of any model from pyOD library, some of which may require installation of additional packages (
see [pyOD docs](https://pyod.readthedocs.io/en/latest/).

### Citing pyCLAD

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


