import numpy as np
from numpy.typing import NDArray

from common.variable import Variable


class BCEWithLogits:
    """
    Binary Cross Entropy with Logit Loss
    """

    def bce_withlogits(self, input: Variable) -> Variable:
        """
        Binary Cross Entropy with Logit Loss
        x: Input of the function, from previous hidden layer.
        y: Labels.
        Return: Variable.
        """

        def sigmoid_function(x: Variable) -> NDArray:
            """
            Sigmoid Function.
            x: Variable. Derivied from previous hidden layer.
            Return: NDArray
            """
            return 1 / (1 + np.exp(-x.value))

        def calculate(x: Variable, y: NDArray):
            return -(y * np.log(sigmoid_function(x)))

        def backward(x: Variable, y: NDArray):
            return calculate(x, y) - y
