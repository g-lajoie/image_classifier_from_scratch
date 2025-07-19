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
        ind_var: Optional[Params] = None,
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
        self._ind_vars = ind_var if isinstance(ind_var, Params) else None
        self._dep_vars = None
        self.weights = Params(None, "Weights")
        self.bias = Params(None, "Bias")

        # Graph Variables
        self._u_out = u_out
        self._parent_layer = parent_layer
        self._next_layer = next_layer

    @property
    def param_dict(self) -> dict[str, Params]:
        return {"weights": self.weights, "ind_var": self.inp, "bias": self.bias}

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
        pass
