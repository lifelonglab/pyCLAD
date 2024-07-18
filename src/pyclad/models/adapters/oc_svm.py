import numpy as np
from sklearn.svm import OneClassSVM

from pyclad.models.adapters.utils import adjust_predictions
from pyclad.models.model_base import Model


class OCSVMAdapter(Model):
    def __init__(self, nu=0.1, gamma=0.1, kernel="rbf"):
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)

    def learn(self, data: np.ndarray):
        self.model.fit(data)

    def predict(self, data: np.ndarray):
        return adjust_predictions(self.model.predict(data)), -self.model.score_samples(data)

    def name(self) -> str:
        return "OneClassSVM"

    def additional_info(self):
        return {"nu": self.nu, "gamma": self.gamma, "kernel": self.kernel}
