import numpy as np
import numpy.typing as NDArray

from image_classifier import NeuralNetwork
from image_classifier.common import Param

from .base_loss_function import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy
    """

    def __init__(self, nn: NeuralNetwork):
        self.input = Param(
            np.zeros(nn.output.shape, dtype=np.float32), label="loss function"
        )

        self.nn = nn
        self.logits: np.ndarray = self.nn.output
        self.y = np.zeros([1, 1], dtype=np.float32)

    @property
    def param_dict(self, *args, **kwargs) -> dict[str, Param]:
        return {self.input.label: self.input}

    def calculate(self) -> np.ndarray:
        """
        Categorical Cross Entropy function.
        """

        logits = self.nn.output

        # Calculations
        m = np.max(logits, axis=1, keepdims=True)
        self.input.value = m + np.log(np.sum(np.exp(logits - m)))

        # Connecting to a parent layer
        self.parent_layer = self.nn.layers[-1]

        return self.input.value

    def backward(self, y_true: np.ndarray):
        """
        Derivative for the Categorical Cross Entropy Function
        """
        # Calculate Gradient
        self.input.grad = self.softmax(self.nn.output) - y_true

        current_layer_grad = self.input.grad

        while self.parent_layer:
            self.parent_layer.backward(current_layer_grad)
            current_layer_grad = self.parent_layer.X.grad

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Softmax function
        """

        return np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
