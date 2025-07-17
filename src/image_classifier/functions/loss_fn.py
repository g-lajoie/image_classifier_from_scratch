from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from common.variable import Variable
from utils.type_helpers import to_ndarry, to_variable


class LossFunction(ABC):
    """
    Interface for activaition class.
    """

    def function(self):
        raise NotImplementedError("The function method has not been implemented.")


class BCEWithLogits(LossFunction):
    """
    Binary Cross Entropy with Logit Loss
    """

    def forward(self, X: Variable | NDArray, y: NDArray) -> Variable:
        """
        Binary Cross Entropy with Logit Loss

        Arguments
            x: Input of the function, from previous hidden layer.
            y: Labels.

        Return: Variable.
        """

        X = to_ndarry(X)

        return to_variable(-(y * np.log(self.sigmoid_function(X))), "BCELogitLoss")

    def sigmoid_function(self, x: NDArray) -> NDArray:
        """
        Sigmoid Function.
        x: Variable. Derivied from previous hidden layer.
        Return: NDArray
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, x: Variable, y: NDArray):
        return self.forward(x, y) - y
