from typing import Optional

import numpy as np
from numpy.typing import NDArray

from common.variable import Variable
from src.activation_functions.base_activation_function import ActivationFunction
from src.weights_initialization import WeightsInitializer
from utils.type_helpers import to_ndarry, to_variable


class LinearLayer:
    """
    The linear (dense) layer of a neural network

    Attributes:
        activation_function: Activation function for linear layer.
        weight_init: Weight Initalizer [Xavier, He] for Weights Matrix initialization.
        u_out [Optional]: Number of untis the layer will output
    """

    def __init__(
        self,
        activation_function: ActivationFunction,
        weight_init: WeightsInitializer,
        u_out: int,
        *args,
        **kwargs,
    ):
        self.activation_function = activation_function
        self.weights_init = weight_init
        self.u_out = u_out

    def forward(self, X: Variable | NDArray) -> np.ndarray:
        """
        Calculates the Linear Layer, to be used in the forward pass.

        Attributes:
            X: Data for the linear layer.
        """

        # Convert X to NDArray
        X = to_ndarry(X)

        W = self.weights_init.init_weights(X, self.u_out)  # Weight matrix
        b = np.zeros(
            W.shape[1],
        )  # bias vector.

        return np.dot(X, W) + b
