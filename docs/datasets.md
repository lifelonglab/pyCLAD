# Data

### Overview 
pyCLAD supports continual anomaly detection datasets built as a sequence of concepts.

A **Concept** is described by a *name*, *data*, and *labels*. 

We provide a loader for datasets saved in numpy format, which returns a **ConceptsDataset** object that organizes concepts' data in two lists: *train* and *test*. 

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


### Continual Scenario Extraction
We devised a simple algorithm to extract continual learning datasets in the format supported by pyCLAD, based on any tabular anomaly detection dataset. 
The implementation can be found [here](https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios). More details can be found in [1]. 

The main steps of the algorithm are:

- Create concepts for the normal class through a concept creation function $\phi$. 

- Create concepts for the anomaly class through anomalous concepts creation function $\gamma$

- For each normal concept $C_{N_i}$, select a corresponding anomaly concept $C_{A_j}$ using a function $\lambda$

- The combination of $C_{N_i}$ and $C_{A_j}$ is a concept added to the continual scenario

- The algorithm returns the resulting scenario as a sequence of concepts, each of which may need to be separated into training and evaluation data depending on the learning settings, e.g., unsupervised or semi-supervised.

In our implementation, the concept creation functions $\phi$ and $\gamma$ are realized through the k-Means clustering algorithm, whereas the concept assignment $\lambda$ supports the following options:

- **CC**: clustered anomaly concepts assigned to the closest normal concept

- **CR**: clustered anomaly concepts assigned randomly to normal concepts

- **R**: anomalies randomly assigned to normal concepts

Alternative scenarios can be designed by customizing $\phi$, $\gamma$, and $\lambda$.


### References 

[1] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.
