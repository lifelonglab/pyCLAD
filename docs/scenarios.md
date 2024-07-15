# Scenarios

### Overview 

To differentiate continual/lifelong anomaly detection scenarios, we define *concept* as a self-consistent behavior of the normal class, alongside the specific anomalies occurring with it.
A concept can correspond to a new distribution, change of a performed activity, or a new state of the environment, depending on the specific analytical context [1].

Scenarios natively supported in pyCLAD are: 

- *Concept-aware*: Known concept identifier and concept boundaries.

- *Concept-incremental*: Unknown concept identifier but known concept boundaries.

- *Concept-agnostic*: Unknown concept identifier and concept boundaries.

New scenarios can be devised to account for the limitations of previously existing ones. One example is the explicit consideration of the temporal dimension in online learning scenarios.

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