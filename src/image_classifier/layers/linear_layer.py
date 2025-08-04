import logging
from types import NoneType
from typing import Optional, cast

import numpy as np
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
        inp: Optional[Param] = None,
        u_out: Optional[int] = None,
        parent_layer: Optional[Layer] = None,
        next_layer: Optional[Layer] = None,
        *args,
        **kwargs,
    ):
        # Layer Variables
        self.weights: Optional[Param] = None
        self.bias: Optional[Param] = None

        # Graph Variables
        self._u_out = u_out
        self._parent_layer = parent_layer
        self._next_layer = next_layer

    @property
    def param_dict(self) -> dict[str, Param]:
        if (self.weights is None) or (self._inp is None) or (self.bias is None):
            logger.error("The Weights, Input(X), or Bias parameters have not been set")
            raise ValueError("The variables: weights, _inp, or bias cannot be None")

        return {"weights": self.weights, "X": self.inp, "bias": self.bias}

    def forward(self):
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        if self.weight_init_method is None:
            logger.error("weight_init attribute is required")
            raise

        if self.weights is None:
            self.weights = Param(
                self.weight_init_method.init_weights(self.inp, self.u_out),
                "Weight",
            )

        if self.bias is None:
            self.bias = Param(np.zeros(self.weights.shape[-1]), "bias vector")

        return np.dot(self.inp, self.weights) + self.bias

    def backward(self):
        """
        Deriviate of the linear layer w.r.t eac parameter
        """

        # Calculate the gradient
        if (self.weights is None) or (self.inp is None) or (self.bias is None):
            logger.error(
                "At least one of the parameters weights, bias, and inp(X) is None."
            )
            raise ValueError("The params weights, bias, and inp cannot be None")

        d_weights = self.inp.value
        d_inp = self.weights.value
        d_bias = np.array(1)

        # Type and Value Checks
        if self.child_layer is None:
            self.weights.grad = d_weights
            self.inp.grad = d_inp
            self.bias.grad = d_bias

        else:
            # Partial Deriative of the child layer.
            d_child_layer_grad = self.child_layer.inp.grad

            # Update Grad
            self.weights.grad = (
                reshape_for_matmul(d_weights, d_child_layer_grad) @ d_child_layer_grad
            )
            self.inp.grad = (
                reshape_for_matmul(d_inp, d_child_layer_grad) @ d_child_layer_grad
            )
            self.bias.grad = np.sum(d_bias, axis=0)
            self.inp.grad = (
                reshape_for_matmul(d_inp, d_child_layer_grad) @ d_child_layer_grad
            )
            self.bias.grad = np.sum(d_bias, axis=0)
