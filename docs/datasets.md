# Data

### Overview

As for now, pyCLAD supports continual anomaly detection datasets built as a sequence of concepts.

A **Concept** is described by a *name*, *data*, and *labels*.

### Built-in Datasets

There are multiple datasets already available in the pyCLAD library. They are stored
on [huggingface](https://huggingface.co/lifelonglab) and can be
automatically downloaded by using the proper class.

As for now, we have:

- **UNSW-NB15** - available through class `UnswDataset`.
- **NSL-KDD** - available through class `NslKddDataset`.
- **Wind Energy** - available through class `WindEnergyDataset`.
- **Energy Plants** - available through class `EnergyPlantsDataset`.

You can easily use any of them by simply importing the class and creating an instance of it. For example:

```python
from pyclad.datasets import UnswDataset

unsw_dataset = UnswDataset(dataset_type='random_anomalies')
```

`dataset_type` can be one of the following: `random_anomalies`, `clustered_with_closest_assignment`, and
`clustered_with_random_assignment`. See more details in the section below describing the continual scenario extraction.

### Continual Scenario Extraction

We devised a simple algorithm to extract continual learning datasets in the format supported by pyCLAD, based on any
tabular anomaly detection dataset.
The implementation can be found [here](https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios). More
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

We provide a loader for datasets saved in numpy format, which returns a **ConceptsDataset** object that organizes
concepts' data in two lists: *train* and *test*.

Its implementation is shown below as a code example.

### Code Example

    def read_dataset_from_npy(filepath: pathlib.Path, dataset_name: str) -> ConceptsDataset:
        data = np.load(str(filepath), allow_pickle=True)
    
        train_concepts = []
        test_concepts = []
    
        for c in data:
            train_concepts.append(Concept(name=c["name"], data=c["train_data"], labels=None))
            if "test_data" in c and len(c["test_data"]) > 0:
                test_data = c["test_data"]
                test_labels = c["test_labels"]
                test_concepts.append(Concept(name=c["name"], data=test_data, labels=test_labels))
    
        return ConceptsDataset(name=dataset_name, train_concepts=train_concepts, test_concepts=test_concepts)


### References

[1] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection:
New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.
