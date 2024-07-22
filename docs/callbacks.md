# Callbacks

### Overview 

Callbacks allow to effectively log useful information in pyCLAD during key checkpoints (*before* and *after*) of the execution workflow: *scenario*, *training*, and *evaluation*.

This information is quite useful for evaluating experimental outcomes (e.g. performance metrics, execution time).

New callbacks can be defined by implementing one or more of the abstract methods in the **Callback** class (e.g. *after_training*, *before_evaluation*, etc.). 
Multiple callbacks are supported during an execution and can be provided as a list to a scenario execution method. Within the scenario, multiple callbacks are processed as **CallbackComposite** object, where the above-mentioned methods are called for each single callback. 

In the following code example, a concept incremental scenario is run considering two callbacks to monitor model performance (*MatrixMetricEvaluationCallback*) and execution time (*TimeEvaluationCallback*).

### Code Example
    from pyclad.callbacks.evaluation.matrix_evaluation import MatrixMetricEvaluationCallback
    ...
    callbacks = [
        MatrixMetricEvaluationCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverageAcrossLearnedConcepts(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback()
    ]
    concept_incremental_scenario(data_loader, strategy=strategy, callbacks=callbacks)
