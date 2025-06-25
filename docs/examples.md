# Examples

In the repository [link](https://github.com/lifelonglab/pyCLAD/tree/main/examples) you can find multiple examples of how
to use the library.

The few most prominent examples are:

1. [Concept-Agnostic Example](https://github.com/lifelonglab/pyCLAD/blob/main/examples/concept_agnostic_example.py) -
   This example show how to create a simple dataset with 3 concepts and carry out a concept agnostic scenario with
   CumulativeStrategy and OneCLassSVM model. Leveraged data is randomly generated to show how to create
   a `ConceptDataset` using any numpy data.
2. [Concept-Incremental Example](https://github.com/lifelonglab/pyCLAD/blob/main/examples/concept_incremental_example.py) -
   This example show how to create a simple dataset with 4 concepts and carry out a concept incremental scenario with
   CumulativeStrategy and IsolationForest model. Leveraged data comes from different normal distributions to showcase
   forgetting and knowledge retention.
3. [Concept-Aware Example](https://github.com/lifelonglab/pyCLAD/blob/main/examples/concept_aware_example.py) - This
   example showcase how to run a concept aware scenario using the NSL-KDD dataset
   stored in `resources` directory. The scenario includes IsolationForestAdapter model and Replay strategy.
4. [UNSW Dataset Example](https://github.com/lifelonglab/pyCLAD/blob/main/examples/unsw_dataset_example.py) - This
   example showcase how to run a concept aware scenario using the UNSW dataset adopted to continual anomaly
   detection. The scenario includes VAE model and Replay strategy.
5. [Energy Dataset Example](https://github.com/lifelonglab/pyCLAD/blob/main/examples/energy_dataset_example.py) - This
   example showcase how to run a concept aware scenario using the Energy dataset adopted to continual anomaly
   detection. The scenario includes LocalOutlierFactor model and Cumulative strategy.
6. [Plot Heatmap Example](https://github.com/lifelonglab/pyCLAD/blob/main/examples/plot_heatmap_example.py) - This
   example showcases how to generate a ROC-AUC heatmap for the results of a concept aware scenario
7. [Models examples](https://github.com/lifelonglab/pyCLAD/tree/main/examples/models) - This directory contains multiple
   examples of how to use the models available in pyCLAD. Each example is a standalone script that demonstrates how to
   use a specific model with a specific strategy and dataset.