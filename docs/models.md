# Models

### Overview 

A model can be added to the library by implementing a model adapter class, subclass of <code>Model</code>, overriding three abstract methods: <code>name</code>, <code>fit</code>, and <code>predict</code>.

A Model class also inherits the abstract method <code>info</code> from <code>InfoProvider</code>, which allows for logging useful information pertaining to the model's configuration as a dictionary.

Although pyCLAD is not restricted to specific models, one-class learning models are the most common in practice.
Models can be built, for instance, based on SkLearn, PyOD, or PyTorch base classes.  
In this case, the <code>fit</code> and <code>predict</code> methods can be wrappers of the <code>fit</code> and <code>predict</code> methods in the respective base model classes.


### PyTorch example
Here is an example of how to implement an Autoencoder model using PyTorch modules:

    class Autoencoder(Model):
        def __init__(
            self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-2, threshold: float = 0.5, epochs: int = 20
        ):
            self.module = AutoencoderModule(encoder, decoder, lr)
            self.threshold = threshold
            self.epochs = epochs
    
        def fit(self, data: np.ndarray):
            dataset = TensorDataset(torch.Tensor(data))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            trainer = pl.Trainer(max_epochs=self.epochs)
            trainer.fit(self.module, dataloader)
    
        def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
            x_hat = self.module(torch.Tensor(data)).detach()
            rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=1)
    
            binary_predictions = (rec_error > self.threshold).astype(int)
            return binary_predictions, rec_error
    
        def name(self) -> str:
            return "Autoencoder"
    
        def additional_info(self):
            return {
                "threshold": self.threshold,
                "encoder": str(self.module.encoder),
                "decoder": str(self.module.decoder),
                "lr": self.module.lr,
                "epochs": self.epochs,
            }

In the current version, we provide three pre-implemented Autoencoder architectures: <code>Autoencoder</code>, <code>TemporalAutoencoder</code>, and <code>VariationalTemporalAutoencoder</code>.
Each can support different <code>encoder</code> and <code>decoder</code> layers implemented as PyTorch modules.

We provide <code>GRUEncoder</code>, <code>LSTMEncoder</code>, <code>TCNEncoder</code>, and their decoder counterparts.
Based on these, custom architectures can be built:

    encoder = LSTMEncoder(_encoder_layers, seq_len=config.seq_len)
    decoder = LSTMDecoder(_decoder_layers, seq_len=config.seq_len)
    autoencoder = TemporalAutoencoder(encoder=encoder, decoder=decoder, epochs=5, seq_len=config.seq_len)

Additional examples can be found in [examples/models](https://github.com/lifelonglab/pyCLAD/tree/main/examples).


### PyOD example
Here is an example of how to use our <code>PyODAdapter</code> to implement existing PyOD models:

    class PyODAdapter(Model):
        def __init__(self, model: BaseDetector, model_name: str):
            self._model = model
            self._model_name = model_name
    
        def fit(self, data: np.ndarray):
            self._model.fit(data)
    
        def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
            return self._model.predict(data), self._model.decision_function(data)
    
        def name(self) -> str:
            return self._model_name
    
        def additional_info(self):
            return self._model.get_params()
    
Here is an adapter implementation for Isolation Forest, a popular one-class tree-based ensemble for anomaly detection.
The implementation is based on the general <code>PyODAdapter</code> class: 

    class IsolationForestAdapter(PyODAdapter):
        def __init__(self, **kwargs):
            super().__init__(model_name="IsolationForest", model=IForest(**kwargs))


Isolation Forest [1] uses a group of tree-based models and calculates an isolation score for every data instance. 
The average distance from the tree's root to the leaf (data instance), i.e., the number of partitions required to reach the instance, is used to compute the anomaly score.

Considering that more noticeable variations in values correspond to shorter paths in the tree, this information is used to detect anomalies from normal instances. 

In custom scenarios, the <code>fit</code> method can entail any arbitrarily complex model training approach, whereas <code>predict</code> could implement a post-processing logic (e.g. from continuous raw scores to binary class outcome, or class label conversion), as done in the example.

Using a similar approach, additional PyOD models can be easily supported. For instance, Local Outlier Factor (LOF) [2]:

    class LocalOutlierFactorAdapter(PyODAdapter):
        def __init__(self, **kwargs):
            super().__init__(model_name="LOF", model=LOF(novelty=True, **kwargs))

### References
 
[1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008, December). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE.

[2] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000, May). LOF: identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD international conference on Management of data* (pp. 93-104).
