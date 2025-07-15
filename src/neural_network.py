import numpy as np


class NeuralNetwork:

    def __init__(self, activation_function, k1: int, *args, **kwargs):
        """
        activation_function: Activation function.
        k1: [int] Hidden units for 1st layer
        ks: [list[int]] Additional hidden units.
        """

        self.activation_function = activation_function
        self.k1 = k1
        self.ks: list[int] = [val for k, val in kwargs.items() if k.startswith("k")]

    def foward_pass(self, X: np.ndarray, weights_initialization_method) -> np.ndarray:
        """
        Forward pass for the neural network.

        X: (shape: n, p) input for neural network.
        """

        parameters: int = X.shape[1]

        # Weight Matrix
        W = np.random.randn(self.k1, parameters)
        b = np.random.randn(self.k1, 1)
