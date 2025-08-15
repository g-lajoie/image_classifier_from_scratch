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
        data: Either initial data or data from previous layers.
        activation_function: Activation function for linear layer.
        weight_init: Weight Initalizer [Xavier, He] for Weights Matrix initialization.
        u_out [Optional]: Number of untis the layer will output
    """

    def __init__(
        self,
        weight_init_method: WeightInitializationMethod,
        units: int,
        label: str,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Layer Variables

        self.weight_init_method: WeightInitializationMethod = weight_init_method
        self.units = units
        self.label = label
        self.weights: Optional[Param] = None
        self.bias: Optional[Param] = None

    def __repr__(self):
        return f"LinearLayer {__name__} output_units:{self.units}"

    @property
    def param_dict(self) -> dict[str, Param | None]:
        return {
            f"{self.label}_weights": self.weights,
            f"{self.label}_X": self.X,
            f"{self.label}_bias": self.bias,
        }

    def forward(self, X: np.ndarray) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        if isinstance(X, np.ndarray):
            self.X: Param = Param(X, "X")

        # Intitialize Weights & Biases
        if self.weights is None:
            self.weights = Param(
                self.weight_init_method.init_weights(self.X, self.units),
                f"Weight: {self.label}",
                grad=np.zeros_like(self.X.value),
            )

        if self.bias is None:
            self.bias = Param(
                np.zeros(self.weights.shape[-1]),
                f"Bias: {self.label}",
                grad=np.ones((1, self.units), dtype=np.float32),
            )

        # Initialize Input Matrix Grad
        self.X.grad = np.zeros_like(self.weights.value)

        # Calcuations
        if self.X.value.shape[-1] != self.weights.value.shape[0]:
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
        self.weights.grad = reshape_for_matmul(self.X.value, previous_layer_grad)
        self.X.grad = reshape_for_matmul(self.X.value, previous_layer_grad)
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
