import logging
from types import NoneType
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.enums import WeightInitMethod
from image_classifier.common.parameters import Params
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.layers.weights_initialization import (
    RandomInitializer,
    ScaledInitializer,
    WeightsInitializer,
)
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
        inp: Optional[Params] = None,
        weight_init_method: WeightInitMethod | None = None,
        u_out: Optional[int] = None,
        parent_layer: Optional[Layer] = None,
        next_layer: Optional[Layer] = None,
        *args,
        **kwargs,
    ):
        # Weight Intialization Method
        self.weight_init = None
        self.weight_init_method: WeightInitMethod | None = weight_init_method

        if weight_init_method == WeightInitMethod.RANDOM:
            self.weight_init = RandomInitializer()

        if (
            weight_init_method == WeightInitMethod.XAVIER
            or weight_init_method == WeightInitMethod.HE
        ):
            self.weight_init = ScaledInitializer(weight_init_method)

        # Layer Variables
        self._inp = inp if isinstance(inp, Params) else None
        self._output = None
        self.weights = Params(None, "Weights")
        self.bias = Params(None, "Bias")

        # Graph Variables
        self._u_out = u_out
        self._parent_layer = parent_layer
        self._next_layer = next_layer

    @property
    def param_dict(self) -> dict[str, Params]:
        return {"weights": self.weights, "X": self.inp, "bias": self.bias}

    def forward(self) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        if self.weight_init is None:
            logger.error("weight_init attribute is required")
            raise

        self.weights = Params(
            self.weight_init.init_weights(self.inp, self.u_out),
            "Weight",
        )

        self.bias = Params(np.zeros(self.weights.shape[-1]), "bias vector")

        return np.dot(self.inp, self.weights) + self.bias

    def backward(self):
        """
        Deriviate of the linear layer w.r.t eac parameter
        """

        # Calculate the gradient
        for v in self.param_dict.values():
            if v.value is None:
                raise ValueError(f"The value for {v.label} is none")

        d_weights = cast(NDArray, self.inp.value)
        d_inp = cast(NDArray, self.weights.value)
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
