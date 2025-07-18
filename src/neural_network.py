import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable
from image_classifier.functions.activiation import RELU
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.layers import LinearLayer
from image_classifier.layers.base_layers import Layers

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Neural Network Model.

    Attributes
        data NDArray: Input data for neural network
        layers Sequence[Layers]: Layers for neural network. Layers will be evaluated in sequential order.
    """

    def __init__(
        self, data: NDArray | None = None, *layers: Sequence[Layers] | None, **kwargs
    ):
        self._data = data
        self.layers = layers

    @property
    def data(self):
        """
        Train, Validation, or Test dataset, loaded externally.
        """
        return self._data

    @data.setter
    def data(self, new_data_value):
        """
        Setter function for data property.
        """
        if isinstance(new_data_value, np.ndarray):
            self._data = new_data_value

        else:
            logger.error(
                "The data value must type<NDArray>, got %s",
                new_data_value,
                exc_info=True,
            )

    def forward(self):
        """
        Defines the forward pass for the neural network model.
        """
