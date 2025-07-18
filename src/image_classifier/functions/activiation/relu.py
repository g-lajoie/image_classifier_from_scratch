import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable
from image_classifier.layers.base_layers import Layers

from .base_activation_function import ActivationFunction

logger = logging.getLogger(__name__)


class RELU(ActivationFunction, Layers):
    """
    ReLU: Rectified Linear Unit
    """

    def __init__(self, data=None):
        self._data = data

    @property
    def data(self) -> NDArray | None:
        return self._data

    @data.setter
    def data(self, new_data_value) -> NDArray | None:

        if isinstance(new_data_value, np.ndarray):
            return new_data_value
        else:
            logger.error(
                "Data must be of type<NDArray>, got %s", new_data_value, exc_info=True
            )
            raise

    def forward(self):
        """
        Caclulates the ReLU function.
        Calculation is done on the value attribute of the Variable object.

        Attributes:
            _in: Varable.
        """
        return np.maximum(0, self.data)

    def backward(self, _in: Variable):
        """
        Calculate the back propgation for the RELU funcion.
        Calculation is done on the value attribute of the Variable object.

        Attributes:
            _in: Variable

        Return: Variable
        """
        return (_in.value > 0).astype(float)
