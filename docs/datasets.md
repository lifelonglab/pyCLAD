# Datasets

### Overview

A dataset (`ConceptsDataset`) is a list of train concepts and a list of test concepts. A `Concept` has a *name*,
*data* and *labels*, with `0` for a normal sample and `1` for an anomaly; a test concept shares its name with the
train concept it belongs to. The order of the train concepts is the order in which they are learned. See
[Core concepts](learn/concepts.md#concepts-and-datasets).

[Using built-in datasets and step schedules](guides/datasets.md) goes through running an experiment on a benchmark.

### Built-in Datasets

pyCLAD ships these benchmarks, hosted on [Hugging Face](https://huggingface.co/lifelonglab):

- **UNSW-NB15** (`UnswDataset`), **NSL-KDD** (`NslKddDataset`), **Wind Energy** (`WindEnergyDataset`) and
  **Energy Plants** (`EnergyPlantsDataset`), built with the continual scenario extraction described below.
- The tabular CAD benchmarks: **CAD-CICIDS2017** (`CadCicids2017Dataset`), **CAD-CICIDS2018**
  (`CadCicids2018Dataset`), **CAD-CICUNSW** (`CadCicunswDataset`), **CAD-TCM** (`CadTcmDataset`),
  **CAD-MiniBooNe** (`CadMiniBooNeDataset`), **CAD-Scania** (`CadScaniaDataset`), **MCAD-CIC-3x1**
  (`McadCic3x1Dataset`) and **MCAD-CIC-3xN** (`McadCic3xNDataset`).

All of them are importable from `pyclad.data.datasets`. They are downloaded and cached on first use; every class
takes a `cache_dir` argument to choose where (by default, the Hugging Face cache). The download needs the `datasets`
package, which `pip install pyclad` already includes.

```python
--8<-- "docs/snippets/longrun/builtin_datasets.py:unsw"
```

`dataset_type` can be one of the following: `random_anomalies`, `clustered_with_closest_assignment`, and
`clustered_with_random_assignment`; see "Continual Scenario Extraction" below.

The CAD/MCAD benchmarks (from the [tabular CAD benchmarks
collection](https://huggingface.co/collections/lifelonglab/tabular-cad-benchmarks), described in [2]) share a common base class,
`TabularCadDataset`, and instead take an `ordering` parameter that selects one of the predefined task sequences
shipped with each dataset in its `orderings.json`:

```python
--8<-- "docs/snippets/longrun/builtin_datasets.py:cad"
```

The ordering is part of the experiment: report it with the results, and compare methods on the same ordering.


### Step Schedule (Grouping)

By default every concept is its own training step. CDAD-style continual anomaly detection
benchmarks instead merge consecutive concepts into larger training steps, while keeping
evaluation per concept. `pyclad.data.grouping` implements this with a compact notation, where
tokens are separated by `-` and `NxM` repeats `N` `M` times:

| Spec | Parsed | Meaning (15 concepts) |
|---|---|---|
| `"14-1"` | `[14, 1]` | one large base step, then one new concept |
| `"10-5"` | `[10, 5]` | two steps |
| `"3x5"` | `[3, 3, 3, 3, 3]` | five steps of three concepts each |
| `"10-1x5"` | `[10, 1, 1, 1, 1, 1]` | a base of 10, then five single-concept steps |

```python
--8<-- "docs/snippets/datasets/step_schedule.py:schedule"
```

The schedule must sum to the number of train concepts. Train concepts are always merged;
merging preserves the concrete concept class, so a `VisionConcept` keeps its `masks` aligned
with `data`. Test concepts stay per category unless `group_test=True`, which requires them to
align 1:1 with the train concepts and describes a different experiment -- do not mix results
from the two in one table.

Because training now takes $T$ steps while evaluation still covers $N$ concepts, the resulting
metric matrix is rectangular ($T \times N$, $T < N$). See
[Metrics](metrics.md#rectangular-matrices-and-step-schedules) for the metrics that can read such a matrix.

Grouping merges *consecutive* concepts, so the dataset must already be in its final order:
**ordering, then grouping, then first_seen_step**. The grouped dataset provides `first_seen_step`, the
training step that first includes each concept, which the schedule-aware metrics need.

### Continual Scenario Extraction

We devised a simple algorithm to extract continual learning datasets in the format supported by pyCLAD from any
tabular anomaly detection dataset. The implementation can be found [here](https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios). More
details can be found in [1].

The main steps of the algorithm are:

- Create concepts for the normal class through a concept creation function $\phi$.

- Create concepts for the anomaly class through anomalous concepts creation function $\gamma$

- For each normal concept $C_{N_i}$, select a corresponding anomaly concept $C_{A_j}$ using a function $\lambda$

- The combination of $C_{N_i}$ and $C_{A_j}$ is a concept added to the continual scenario

- The algorithm returns the resulting scenario as a sequence of concepts, each of which may need to be separated into
  training and evaluation data depending on the learning settings, e.g., unsupervised or semi-supervised.

In our implementation, the concept creation functions $\phi$ and $\gamma$ are realized through the k-Means clustering
algorithm, whereas the concept assignment $\lambda$ supports the following options:

- **CC** (`clustered_with_closest_assignment`): clustered anomaly concepts assigned to the closest normal concept

- **CR** (`clustered_with_random_assignment`): clustered anomaly concepts assigned randomly to normal concepts

- **R** (`random_anomalies`): anomalies randomly assigned to normal concepts

Alternative scenarios can be designed by customizing $\phi$, $\gamma$, and $\lambda$.

### Loading dataset from numpy format produced by the continual scenario extraction algorithm

`read_dataset_from_npy` (in `pyclad.data.readers`) loads a scenario saved by the extraction algorithm and returns a
`ConceptsDataset`. The file holds an array with one dictionary per concept, in the continual order:

```python
--8<-- "docs/snippets/datasets/npy_format.py:format"
```

```python
--8<-- "docs/snippets/datasets/npy_format.py:read"
```

A concept without test data gets no test concept, so the test list can be shorter than the train list.

### Using your own data

Any data can be used by splitting it into concepts and building a `ConceptsDataset(name, train_concepts,
test_concepts)`, as in [your first experiment](getting_started.md). Give test concepts the same names as the train
concepts they belong to. Unsupervised strategies treat all training data as normal, so if your training data contains
anomalies, state it when you report results.

### References

[1] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection:
New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.

[2] Faber, K., Smendowski, M., & Corizzo, R. (2026). Towards Principled Continual Anomaly Detection: A Systematic
Framework and Benchmark Scenarios. [arXiv:2607.18289](https://arxiv.org/abs/2607.18289).
