# Getting Started

### Overview 

A typical execution requires:

- Loading a continual anomaly detection *dataset*;
 
- Defining the type of *scenario*;

- Define the *strategy* and the *model* subject to evaluation;

- Define optional *callbacks* to log specific information during the execution.

More detailed information about each of these components can be found in specific sections of the documentation.

### Code Example
    
    if __name__ == "__main__":
        data_loader = read_dataset_from_npy(
            pathlib.Path("resources/nsl-kdd_random_anomalies_5_concepts_1000_per_cluster.npy"), dataset_name="NSL-KDD-R"
        )
        strategy = CumulativeStrategy(IsolationForestAdapter())
        callbacks = [
            MatrixMetricEvaluationCallback(
                base_metric=RocAuc(),
                metrics=[ContinualAverageAcrossLearnedConcepts(), BackwardTransfer(), ForwardTransfer()],
            ),
            TimeEvaluationCallback()
        ]
        concept_aware_scenario(data_loader, strategy=strategy, callbacks=callbacks)
    
        output_writer = JsonOutputWriter(pathlib.Path("output.json"))
        output_writer.write([data_loader, strategy, *callbacks])

In this example, experiments on the NSL-KDD dataset are run on a *Concept-aware scenario* considering a *Cumulative strategy* and an *Isolation Forest model*. 

More examples with different types of scenarios can be found in **\examples**. 

An example of output (*output.json*) is shown below, in which continual learning metrics and execution time are reported.
    
    {
        "dataset": {
            "name": "NSL-KDD-R",
            "train_concepts_no": 5,
            "test_concepts_no": 5
        },
        "strategy": {
            "name": "Cumulative",
            "model": "IsolationForest",
            "replay_size": 50
        },
        "matrixMetricEvaluationCallback_ROC-AUC": {
            "ROC-AUC": {
                "metrics": {
                    "ContinualAverageAcrossLearnedConcepts": 0.9275,
                    "BackwardTransfer": -0.028874999999999994,
                    "ForwardTransfer": 0.6545
                },
                "concepts_order": [
                    "Cluster_0",
                    "Cluster_1",
                    "Cluster_2",
                    "Cluster_3",
                    "Cluster_4"
                ],
                "matrix": {
                    "Cluster_0": {
                        "Cluster_0": 0.99375,
                        "Cluster_1": 0.905,
                        "Cluster_2": 0.7225,
                        "Cluster_3": 0.48125,
                        "Cluster_4": 0.4875
                    },
                    ...
                    "Cluster_4": {
                        ...
                    }
                }
            }
        },
        "time_by_concept": {
            "Cluster_0": {
                "train_time": 0.6884689331054688,
                "eval_time": 0.26117944717407227
            },
            ...
            "Cluster_4": {
                "train_time": 1.1764769554138184,
                "eval_time": 0.2475874423980713
            }
        },
        "train_time_total": 4.74727988243103,
        "eval_time_total": 1.2206685543060303
    }

