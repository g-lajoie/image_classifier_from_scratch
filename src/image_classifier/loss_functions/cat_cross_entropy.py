import numpy as np
import numpy.typing as NDArray

from image_classifier import NeuralNetwork
from image_classifier.common import Param

from .base_loss_function import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy
    """

    def __init__(self):
        self.y = np.zeros([1, 1], dtype=np.float32)

    def calculate(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Categorical Cross Entropy function.
        """

        N = logits.shape[0]

        # Calculations
        m = np.max(logits, axis=1, keepdims=True)
        cce = m + np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))
        z_correct = logits[np.arange(N), labels.astype(np.int32)].reshape(-1, 1)

        return -(z_correct - cce)

    def backward(self, logits: np.ndarray, y_true: np.ndarray):
        """
        Derivative for the Categorical Cross Entropy Function
        """
        # Calculate Gradient
        return self.softmax(logits) - y_true

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Softmax function
        """

        N = logits.shape[0]
        m = np.max(logits, axis=1, keepdims=True)

        return np.exp(logits - m) / np.sum(np.exp(logits - m))
