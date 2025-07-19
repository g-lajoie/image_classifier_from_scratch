import numpy as np
import numpy.typing as NDArray

from image_classifier.common import Params
from image_classifier.layers.base_layers import Layer

from .base_loss_function import LossFunction


class CatCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy
    """

    @property
    def param_dict(self) -> dict[str, Params]:
        """
        Dictionary of params
        """

        return {"ind_vars": self.inp}

    def forward(self, X: Params, y_true: np.ndarray, error=1e-8) -> Params:
        """
        Categorical Cross Entropy function.
        """

        y_pred = self.softmax(X)
        return Params(-np.sum(y_pred * np.log(y_pred + error)), "SoftmaxWithCCE")

    def softmax(self, X: Params) -> np.ndarray:
        """
        Softmax function
        """

        return np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)))

    def backward(self, y_true: np.ndarray):
        """
        Deravative for the Categorical Cross Entropy Function
        """

        return self.inp - y_true
