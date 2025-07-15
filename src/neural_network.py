import numpy as np

from src.weights_initialization import WeightsInitializer


class NeuralNetwork:

    def __init__(
        self,
        activation_function,
        weight_init: WeightsInitializer,
        k1: int,
        *args,
        **kwargs
    ):
        """
        activation_function: Activation function.
        k1: [int] Hidden units for 1st layer
        ks: [list[int]] Additional hidden units.
        """

        self.activation_function = activation_function
        self.k1 = k1
        self.ks: list[int] = [val for k, val in kwargs.items() if k.startswith("k")]

        self.weights_init = weight_init

    def linear_layer(self, X: np.ndarray, weights_initialization_method) -> np.ndarray:
        """
        Forward pass for the neural network.

        X: (shape: n, p) input for neural network.
        """

        features: int = X.shape[1]

        W = self.weights_init(features, self.k1)  # Weight matrix
        b = np.random.randn(features, 1)  # bias vector.

        return np.dot(X, W) + b
