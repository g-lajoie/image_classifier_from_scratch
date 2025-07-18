import logging
from types import NoneType
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray
from utils.type_helpers import to_ndarry, to_variable

from image_classifier.common.enums import WeightInitMethod
from image_classifier.common.variable import Variable
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.layers.weights_initialization import (
    RandomInitializer,
    ScaledInitializer,
    WeightsInitializer,
)

from .base_layers import Layers

logger = logging.getLogger(__name__)


class LinearLayer(Layers):
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
        ind_var: Optional[Variable] = None,
        weight_init_method: WeightInitMethod | None = None,
        u_out: Optional[int] = None,
        parent_layer: Optional[Layers] = None,
        next_layer: Optional[Layers] = None,
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
        self._ind_vars = ind_var if isinstance(ind_var, Variable) else None
        self._dep_vars = None
        self.weights = Variable(None, "Weights")
        self.bias = Variable(None, "Bias")

        # Graph Variables
        self._u_out = u_out
        self._parent_layer = parent_layer
        self._next_layer = next_layer

    @property
    def variables(self):
        return [self.weights, self.ind_var, self.bias]

    def forward(self) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        if self.weight_init is None:
            logger.error("weight_init attribute is required")
            raise

        self.weights = Variable(
            self.weight_init.init_weights(self.ind_var, self.u_out),
            "Weight",
        )

        self.bias = Variable(np.zeros(self.weights.shape[-1]), "bias vector")

        return np.dot(self.ind_var, self.weights) + self.bias

    def backward(self):
        pass
