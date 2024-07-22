# Getting Started

### Overview 

To get started, we describe a typical pyCLAD execution step-by-step. 
In the example below, we run experiments on the NSL-KDD dataset on a *Concept-aware scenario* considering a *Cumulative strategy* and an *Isolation Forest model*

A typical execution requires:

- Loading a continual anomaly detection *dataset*. 

Here, we use the numpy pyCLAD data loader (see [Data](data.md)) on the pre-processed NSL-KDD dataset:
 
        data_loader = read_dataset_from_npy(
            pathlib.Path("resources/nsl-kdd_random_anomalies_5_concepts_1000_per_cluster.npy"), 
            dataset_name="NSL-KDD-R"
        )

- Define the *strategy* (see [Strategies](strategies.md)) and the *model* (see [Models](models.md)) subject to evaluation 

        strategy = CumulativeStrategy(IsolationForestAdapter())
 
- Define optional *callbacks* to log specific information during the execution (see [Callbacks](callbacks.md)).

  Here, we log model performance (*MatrixMetricEvaluationCallback*) and execution time (*TimeEvaluationCallback*)

        callbacks = [
            MatrixMetricEvaluationCallback(
                base_metric=RocAuc(),
                metrics=[ContinualAverageAcrossLearnedConcepts(), BackwardTransfer(), ForwardTransfer()],
            ),
            TimeEvaluationCallback()
        ]

- Choose and run a *scenario* (see [Scenarios](scenarios.md))

        concept_aware_scenario(data_loader, strategy=strategy, callbacks=callbacks)

- Save results

        output_writer = JsonOutputWriter(pathlib.Path("output.json"))
        output_writer.write([data_loader, strategy, *callbacks])

More detailed information about each of these components can be found in specific sections of the documentation.

An example of output in JSON format (*output.json*) is shown below, in which continual learning metrics and execution time are reported.
    
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
                    "BackwardTransfer": -0.0288,
                    "ForwardTransfer": 0.6545
                },
                "concepts_order": [
                    "Cluster_0",
                    ...
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
                "train_time": 0.6884,
                "eval_time": 0.2611
            },
            ...
            "Cluster_4": {
                "train_time": 1.1764,
                "eval_time": 0.2475
            }
        },
        "train_time_total": 4.7472,
        "eval_time_total": 1.2206
    }

