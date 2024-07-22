# Overview

**InfoProvider** is an abstract class that provides convenient information log for different classes (Dataset, Model, Strategy, etc.).

Classes must implement the abstract method **info** to output important information that should be saved in results (e.g. names) as a Python dictionary.
For example, this mechanisms ensures that strategies uses the key "strategy" and provides a strategy name in the output. An example of JSON output is the following:

    {
        "strategy": {
            # info:
            "name": "Cumulative",

            # additional info from specific strategies 
            "model": "IsolationForest",
            "replay_size": 50
        }
    }


Moreover, optional information, such as hyperparameters defining a specific configuration of models, strategies, etc can be provided through the **additional_info()** method.

This information can be passed to a **JsonOutputWriter** as a list of **InfoProvider** objects to save information about a pyCLAD execution as a JSON output file.
    

## Code Example: Dataset 
    
    class ConceptsDataset(Dataset):
        ...
    
        def name(self) -> str:
            return self._name
    
        def additional_info(self):
            return {"train_concepts_no": len(self._train_concepts), "test_concepts_no": len(self._test_concepts)}


## Code Example: Model

The *Model* class implements *InfoProvider* to log useful information about specific models. 

All models have a **name** and can provide additional information (e.g. number of estimators in **IsolationForest**) through the **additional_info()** method:
    
    class IsolationForestAdapter(Model):
        ...

        def name(self) -> str:
            return "IsolationForest"
    
        def additional_info(self):
            return {"n_estimators": self.n_estimators, "contamination": self.contamination}



## Code Example: Strategy

The **Strategy** class implements *InfoProvider* to log useful information about specific strategies. 

All strategies have a **name** which is included in the **info** method:

    class Strategy(InfoProvider):
        """Base class for all continual learning strategies."""
    
        @abc.abstractmethod
        def name(self) -> str: ...
    
        def additional_info(self) -> Dict:
            return {}
    
        def info(self) -> Dict[str, Any]:
            return {"strategy": {"name": self.name(), **self.additional_info()}}

Strategies can provide additional information (e.g. replay buffer size) through the **additional_info()** method: 
    
    class ReplayOnlyStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
        ...
    
        def name(self) -> str:
            return "ReplayOnly"
    
        def additional_info(self) -> Dict:
            return {"replay_size": len(self._buffer.data())}