# Callbacks

### Overview 

Callbacks allow to effectively log useful information in pyCLAD during key checkpoints (*before* and *after*) of the execution workflow: *scenario*, *training*, and *evaluation*.

This information is quite useful for evaluating experimental outcomes (e.g. performance metrics, execution time).

New callbacks can be defined by implementing one or more of the abstract methods in the **Callback** class (e.g. *after_training*, *before_evaluation*, etc.). 
Multiple callbacks are supported during an execution and can be provided as a list to a scenario execution method. Within the scenario, multiple callbacks are processed as **CallbackComposite** object, where the above-mentioned methods are called for each single callback. 

Callbacks are also subclass of **InfoProvider**, a utility class that logs information in structured format (Python dictionary).
Multiple callbacks can be passed to a **JsonOutputWriter** as a list of **InfoProvider** objects to save information about a pyCLAD execution as a JSON output file.

In the following code example, a concept incremental scenario is run considering two callbacks to monitor model performance (*MatrixMetricEvaluationCallback*) and execution time (*TimeEvaluationCallback*).

### Code Example
    ...
    callbacks = [
        MatrixMetricEvaluationCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverageAcrossLearnedConcepts(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback()
    ]
    concept_incremental_scenario(data_loader, strategy=strategy, callbacks=callbacks)

