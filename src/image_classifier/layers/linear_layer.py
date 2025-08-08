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
        layer_input: NDArray | Layer,
        units: int,
        label: str,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Layer Variables
        self.layer_input = layer_input
        self.units = units

        if isinstance(self.layer_input, np.ndarray):
            self.X: Param = Param(self.layer_input, "X")

        if isinstance(self.layer_input, Layer):
            self.parent_layer: Layer = self.layer_input

            self.X: Param = Param(
                np.zeros_like(self.parent_layer.X.value, dtype=np.float32),
                label=f"X: {self.label}",
            )

        self.weight_init_method: Optional[WeightInitializationMethod] = None
        self.weights: Optional[Param] = None
        self.bias: Optional[Param] = None

    def __repr__(self):
        return f"LinearLayer {__name__} output_units:{self.units}"

    @property
    def param_dict(self) -> dict[str, Param]:
        if (self.weights is None) or (self.X is None) or (self.bias is None):
            logger.error("The Weights, Input, or Bias parameters have not been set")
            raise ValueError("The variables: weights, _inp, or bias cannot be None")

        return {"weights": self.weights, "X": self.X, "bias": self.bias}

    def forward(self) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        # Pre Calculation Parameters
        if self.weight_init_method is None:
            raise ValueError("weight_init attribute is required")

        if self.units is None:
            raise ValueError("The output units must be set.")

        # Weights & Biases
        if self.weights is None:
            self.weights = Param(
                self.weight_init_method.init_weights(self.X, self.units),
                f"Weight: {self.label}",
            )

        if self.bias is None:
            self.bias = Param(np.zeros(self.weights.shape[-1]), f"Bias: {self.label}")

        if self.X.value.shape[-1] != self.weights.value.shape[0]:
            ValueError(f"Dimension mismatch in Layer{self.label}")

        return self.X.value @ self.weights.value + np.asarray(self.bias.value)

    def backward(self, previous_layer_grad: NDArray):
        """
        Deriviate of the linear layer w.r.t eac parameter
        """

        # Calculate the gradient
        if (self.weights is None) or (self.X is None) or (self.bias is None):
            logger.error(
                "At least one of the parameters weights, bias, and X(X) is None."
            )
            raise ValueError("The params weights, bias, and X cannot be None")

        d_weights = self.X.value
        d_X = self.weights.value
        d_bias = np.array(1)

        # Update Grad
        self.weights.grad = (
            reshape_for_matmul(d_weights, previous_layer_grad) @ previous_layer_grad
        )
        self.X.grad = reshape_for_matmul(d_X, previous_layer_grad) @ previous_layer_grad
        self.bias.grad = np.sum(d_bias, axis=0)

        return self.X.grad
