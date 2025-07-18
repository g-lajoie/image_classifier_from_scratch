import logging
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
        data: Optional[NDArray] = None,
        weight_init_method: WeightInitMethod | None = None,
        next_layer: Optional[Layers] = None,
        *args,
        **kwargs,
    ):
        # Weight Intialization Method
        self.weight_init = None
        self.weight_init_method: WeightInitMethod | None = None

        if weight_init_method == WeightInitMethod.RANDOM:
            self.weight_init = RandomInitializer()

        if (
            weight_init_method == WeightInitMethod.XAVIER
            or weight_init_method == WeightInitMethod.HE
        ):
            self.weight_init = ScaledInitializer(weight_init_method)

        # Layer Variables
        self._ind_vars = Variable(data, "input_var") if data else None
        self._dep_vars = None
        self.weights = None
        self.bias = None

        # Graph Variables
        self._u_out = None
        self.next_layer = next_layer

    @property
    def ind_vars(self) -> Variable:
        """
        Independent variable for linear layer.
        """
        if self._ind_vars is None:
            logger.error("No data provided")

        if not isinstance(self._ind_vars, Variable):
            logger.error(
                "Incorrect type of ind_vars. Expected <Variable> got <%s>",
                type(self.ind_vars),
                exc_info=True,
            )

        return cast(Variable, self._ind_vars)

    @ind_vars.setter
    def ind_vars(self, data) -> Variable:
        """
        Setter function for independent variable.
        """
        self._data = data

    @property
    def dep_vars(self) -> Variable

    @property
    def variables(self):
        return [self.weights, self.data, self.bias]

    @property
    def u_out(self):
        """
        The number of units that
        """

    def forward(self) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        # Convert X to NDArray
        if self.data is None:
            logger.error(
                "No data loaded into this layer, please provide data and rerun the model",
                exc_info=True,
            )
            raise

        if self.weight_init is None:
            logger.error("weight_init attribute is required")
            raise

        # Variables
        self.weights = Variable(
            self.weight_init.init_weights(self.data, self.u_out),
            "Weight",
        )

        b = Variable(np.zeros(W.shape[1]), "bias vector", self.layer_name)

        return np.dot(self.data, W) + b

    def backward(self):
        pass
