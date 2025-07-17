from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from common.variable import Variable


class ActivationFunction(ABC):
    """
    Interface for activaition class.
    """

    def function(self):
        raise NotImplementedError("The function method has not been implemented.")


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
