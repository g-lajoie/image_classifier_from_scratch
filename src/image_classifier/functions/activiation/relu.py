from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable

from .base_activation_function import ActivationFunction


class RELU(ActivationFunction):
    """
    ReLU: Rectified Linear Unit
    """

    def __init__(self):
        pass

    def forward(self, _in: Variable):
        """
        Caclulates the ReLU function.
        Calculation is done on the value attribute of the Variable object.

        Attributes:
            _in: Varable.
        """
        _in.value = np.maximum(0, _in.value)
        return

    def backward(self, _in: Variable):
        """
        Calculate the back propgation for the RELU funcion.
        Calculation is done on the value attribute of the Variable object.

        Attributes:
            _in: Variable

        Return: Variable
        """
        return (_in.value > 0).astype(float)
