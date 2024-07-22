# Models

### Overview 

A model can be added to the library by implementing a model adapter class, subclass of <code>Model</code>, overriding three abstract methods: <code>name</code>, <code>learn</code>, and <code>predict</code>.

A Model class also inherits the abstract method <code>info</code> from <code>InfoProvider</code>, which allows for logging useful information pertaining to the model's configuration as a dictionary.

Although pyCLAD is not restricted to specific models, one-class learning models are the most common in practice.
Models can be built, for instance, based on SkLearn or PyOD base classes.  
In this case, the <code>learn</code> and <code>predict</code> methods can be wrappers of the <code>fit</code> and <code>predict</code> methods in the respective base model classes.

In the current version, we provide an adapter implementation for Isolation Forest, a popular one-class tree-based ensemble for anomaly detection.

### Code example

    import numpy as np
    from sklearn.ensemble import IsolationForest                 
    from pyclad.models.adapters.utils import adjust_predictions  
    from pyclad.models.model_base import Model                   

    class IsolationForestAdapter(Model):
        def __init__(self, n_estimators=100, contamination=0.00001):
            self.n_estimators = n_estimators
            self.contamination = contamination
            self.model = IsolationForest(n_estimators=self.n_estimators, contamination=0.00001)
    
        def learn(self, data: np.ndarray):
            self.model.fit(data)                
    
        def predict(self, data: np.ndarray):
            return adjust_predictions(self.model.predict(data)), -self.model.score_samples(data)
    
        def name(self) -> str:
            return "IsolationForest"


Isolation Forest [1] uses a group of tree-based models and calculates an isolation score for every data instance. 
The average distance from the tree's root to the leaf (data instance), i.e., the number of partitions required to reach the instance, is used to compute the anomaly score. 
For mode details see [IsolationForest in Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

Considering that more noticeable variations in values correspond to shorter paths in the tree, this information is used to detect anomalies from normal instances. 

In custom scenarios, the <code>learn</code> method can entail any arbitrarily complex model training approach, whereas <code>predict</code> could implement a post-processing logic (e.g. from continuous raw scores to binary class outcome, or class label conversion), as done in the example.

### References
 
[1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008, December). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE.

