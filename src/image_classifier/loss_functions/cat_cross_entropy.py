import numpy as np
import numpy.typing as NDArray

from image_classifier.common import Param
from image_classifier.layers.base_layers import Layer

from .base_loss_function import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy
    """

    def __init__(self, inp: Param):
        self.input = inp

    @property
    def param_dict(self) -> dict[str, Param]:
        """
        Dictionary of params
        """

        return {"inp": self.input}

    def calculate(self, y_true: np.ndarray, error=1e-8) -> np.ndarray:
        """
        Categorical Cross Entropy function.
        """

        y_pred = self.softmax(self.input)
        return -np.sum(y_pred * np.log(y_pred + error))

    def backward(self, y_true: np.ndarray):
        """
        Derivative for the Categorical Cross Entropy Function
        """

        self.input.grad = self.input - y_true

    def softmax(self, X: Param) -> np.ndarray:
        """
        Softmax function
        """

        return np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)))
