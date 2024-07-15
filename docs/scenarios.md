# Scenarios

### Overview 

A continual learning **scenario** defines data assumptions that have implications on how *strategies* and *models* behave. 

To differentiate continual anomaly detection scenarios, we define **concept** as a self-consistent behavior of the normal class, alongside the specific anomalies occurring with it.
A concept can correspond to a new distribution, change of a performed activity, or a new state of the environment, depending on the specific analytical context [1].

*Example*: In monitoring human conditions to detect harmful states, the entire normal class can be thought of as a set of concepts: *resting*, *jogging*, and *eating*, each with distinct characteristics.

Scenarios differ based on the availability of:

- **Concept identifier**: A consistent behavior of the normal class describing one specific activity
- **Concept boundary**: Explicit information on whether the currently analyzed concept (a specific activity) has changed.

Scenarios natively supported in pyCLAD are: 

- **Concept-aware**: Known concept identifier and concept boundaries.
This scenario implies that the model is aware of the currently processed activity and its lifespan (at both training and inference time).

- **Concept-incremental**: Unknown concept identifier but known concept boundaries.
This scenario only provides an indication that a change of activity has occurred without any identifying information about the specific activities.

- **Concept-agnostic**: Unknown concept identifier and concept boundaries.
This scenario is more challenging for models/strategies than the previous ones, as it does not provide any supporting information about the current activity being performed and its lifespan.

New scenarios can be designed to account for the limitations of previously existing ones. One example is the explicit consideration of the temporal dimension in online learning scenarios.


### Code example
    def concept_aware_scenario(data_loader: ConceptsDataset, strategy: ConceptAwareStrategy, callbacks: List[Callback]):
        callback_composite = CallbackComposite(callbacks)
    
        for train_concept in data_loader.train_concepts():
            logger.info(f"Starting training on concept {train_concept.name}")
            callback_composite.before_training()
            strategy.learn(data=train_concept.data, concept_id=train_concept.name)
            callback_composite.after_training(learned_concept=train_concept)
    
            for test_concept in data_loader.test_concepts():
                logger.info(f"Starting evaluation of concept {train_concept.name}")
                callback_composite.before_evaluation()
                anomaly_scores, y_predicted = strategy.predict(data=test_concept.data, concept_id=test_concept.name)
                callback_composite.after_evaluation(
                    evaluated_concept=test_concept,
                    y_true=test_concept.labels,
                    y_pred=y_predicted,
                    anomaly_scores=anomaly_scores,
                )
    
        callback_composite.after_scenario()

### References

[1] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.