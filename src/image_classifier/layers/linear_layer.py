import logging
from types import NoneType
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.utils.matrix_calculation_helpers import reshape_for_matmul
from image_classifier.weight_initializers.base_weight_initialization import (
    WeightInitializationMethod,
)

from .base_layers import Layer

logger = logging.getLogger(__name__)


class LinearLayer(Layer):
    """
    The linear (dense) layer of a neural network

    Attributes:
        weight_init: Weight Initalizer [Xavier, He] for Weights Matrix initialization.
        u_out: Number of untis the layer will output
        label: Label for layer of neural network
        shape: Shape of the input.
    """

    def __init__(
        self,
        weight_init_method: WeightInitializationMethod,
        in_features: int,
        out_features: int,
        label: str,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Layer Variables

        self.weight_init_method: WeightInitializationMethod = weight_init_method
        self.in_features = in_features
        self.out_features = out_features
        self.label = label

        # Intialize Weights and Bias
        weights_initializer = self.weight_init_method.init_weights(
            in_features, out_features
        )

        self.weights: Param = Param(
            weights_initializer,
            f"Weight: {self.label}",
            np.zeros([in_features, out_features]),
            (in_features, out_features),
        )

        self.bias: Param = Param(
            np.ones((1, out_features), dtype=np.float64),
            f"Bias: {self.label}",
            np.ones((1, out_features), dtype=np.float64),
            (1, out_features),
        )

    def __repr__(self):
        return f"LinearLayer {__name__}: {self.label}"

    def forward(self, X: np.ndarray) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        if isinstance(X, np.ndarray):
            self.X: Param = Param(
                X, "X", np.zeros_like(X), shape=(X.shape[0], X.shape[1])
            )

        # Calcuations
        if self.X.value.shape[1] != self.weights.value.shape[0]:
            ValueError(f"Dimension mismatch in Layer{self.label}")

        return self.X.value @ self.weights.value + np.asarray(self.bias.value)

    def backward(self, previous_layer_grad: NDArray) -> np.ndarray:
        """
        Deriviate of the linear layer w.r.t eac parameter
        """

        # Calculate the gradient
        if (self.weights is None) or (self.X is None) or (self.bias is None):
            logger.error(
                "At least one of the parameters weights, bias, and X(X) is None."
            )
            raise ValueError("The params weights, bias, and X cannot be None")

        # Update Grad
        self.weights.grad = np.transpose(self.X.value) @ previous_layer_grad
        self.X.grad = previous_layer_grad @ np.transpose(self.weights.value)
        self.bias.grad = np.sum(previous_layer_grad, axis=0, keepdims=True)

        return self.X.grad

    def _zero_grad(self):

        if self.X is None or self.weights is None or self.bias is None:
            raise ValueError(
                "The input (X), weights, and bias parameters cannot be None"
            )

        self.X.grad = np.zeros_like(self.X.grad)
        self.weights.grad = np.zeros_like(self.X.grad)
        self.bias.grad = np.zeros_like(self.X.grad)
