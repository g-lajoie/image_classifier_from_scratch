import numpy as np
import numpy.typing as NDArray

from image_classifier.common import Params
from image_classifier.layers.base_layers import Layer

from .base_loss_function import LossFunction


class CatCrossEntropy(LossFunction, Layer):
    """
    Categorical Cross Entropy
    """

    def forward(self, X: Params, y: np.ndarray, error=1e-8) -> Params:
        """
        Categorical Cross Entropy function.
        """

        y_pred = self.softmax(X)
        return Params(-np.sum(y * np.log(y_pred + error)), "SoftmaxWithCCE")

    def softmax(self, X: Params) -> np.ndarray:
        """
        Softmax function
        """

        return np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)))
