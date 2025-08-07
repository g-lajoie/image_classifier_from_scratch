import logging
from types import NoneType
from typing import Optional, cast

import numpy as np
from common.parameters import Param
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.utils.matrix_calculation_helpers import reshape_for_matmul

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
        input: NDArray | Layer,
        units: int,
        *args,
        **kwargs,
    ):
        if isinstance(input, np.ndarray):
            self.input = Param(input, "input")
            self.units = units

        if isinstance(input, Layer):
            self.input = input.output
            self.units = units
            self.parent_layer = input

        # Layer Variables
        self.weights: Optional[Param] = None
        self.bias: Optional[Param] = None

    def __repr__(self):
        return f"LinearLayer {__name__} output_units:{self.units}"

    @property
    def param_dict(self) -> dict[str, Param]:
        if (self.weights is None) or (self.input is None) or (self.bias is None):
            logger.error("The Weights, Input, or Bias parameters have not been set")
            raise ValueError("The variables: weights, _inp, or bias cannot be None")

        return {"weights": self.weights, "X": self.input, "bias": self.bias}

    def forward(self) -> None:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        if self.weight_init_method is None:
            logger.error("weight_init attribute is required")
            raise

        if self.units is None:
            logger.error("The output units must be set.")
            raise

        if self.weights is None:
            self.weights = Param(
                self.weight_init_method.init_weights(self.input, self.units),
                "Weight",
            )

        if self.bias is None:
            self.bias = Param(np.zeros(self.weights.shape[-1]), "bias vector")

        self.output = np.dot(self.input, self.weights) + self.bias

    def backward(self):
        """
        Deriviate of the linear layer w.r.t eac parameter
        """

        # Calculate the gradient
        if (self.weights is None) or (self.input is None) or (self.bias is None):
            logger.error(
                "At least one of the parameters weights, bias, and input(X) is None."
            )
            raise ValueError("The params weights, bias, and input cannot be None")

        d_weights = self.input.value
        d_input = self.weights.value
        d_bias = np.array(1)

        # Type and Value Checks
        if self.child_layer is None:
            self.weights.grad = d_weights
            self.input.grad = d_input
            self.bias.grad = d_bias

        else:
            # Partial Deriative of the child layer.
            d_child_layer_grad = self.child_layer.input.grad

            # Update Grad
            self.weights.grad = (
                reshape_for_matmul(d_weights, d_child_layer_grad) @ d_child_layer_grad
            )
            self.input.grad = (
                reshape_for_matmul(d_input, d_child_layer_grad) @ d_child_layer_grad
            )
            self.bias.grad = np.sum(d_bias, axis=0)
            self.input.grad = (
                reshape_for_matmul(d_inp, d_child_layer_grad) @ d_child_layer_grad
            )
            self.bias.grad = np.sum(d_bias, axis=0)
