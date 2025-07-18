import logging
from abc import ABC, abstractmethod
from typing import cast

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

    def __init__(self, ind_var=None):
        self._ind_var: Variable | None = ind_var
        self._dep_var: Variable | None = None
        self._

    @property
    def variables(self):
        """
        List of all the variables for the layer
        """

        return

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
