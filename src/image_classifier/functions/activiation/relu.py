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

        Attributes:
            _in: Units from linear layer.
        """
        pass

    def function(self):
        return
