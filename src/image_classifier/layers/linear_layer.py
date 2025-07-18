import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from utils.type_helpers import to_ndarry, to_variable

from image_classifier.common.variable import Variable
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.layers.weights_initialization import WeightsInitializer

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
        weight_init: WeightsInitializer,
        u_out: int,
        data: Optional[NDArray] = None,
        layer_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.weights_init = weight_init
        self.u_out = u_out
        self._data = data
        self.layer_name = layer_name

    @property
    def data(self) -> NDArray | None:
        return self._data

    @data.setter
    def data(self, new_data_value) -> NDArray | None:
        self._data = new_data_value

    def forward(self) -> NDArray:
        """
        Calculates the Linear Layer, to be used in the forward pass.
        """

        # Convert X to NDArray
        if self.data:
            X = to_ndarry(self.data)

        else:
            logger.error(
                "No data loaded into this layer, please provide data %s and rerun the model",
                self.__repr__(),
                exc_info=True,
            )
            raise

        W = Variable(
            self.weights_init.init_weights(X, self.u_out), "Weight", self.layer_name
        )
        b = Variable(np.zeros(W.shape[1]), "bias vector", self.layer_name)

        return np.dot(X, W) + b

    def backward(self):
        pass
