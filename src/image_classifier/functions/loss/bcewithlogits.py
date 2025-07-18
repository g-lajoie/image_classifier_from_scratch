from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable
from image_classifier.utils.type_helpers import to_ndarry, to_variable

from .base_loss_function import LossFunction


class BCEWithLogits(LossFunction):
    """
    Binary Cross Entropy with Logit Loss
    """

    def forward(self, X: Variable, y: NDArray):
        """
        Binary Cross Entropy with Logit Loss

        Arguments
            x: Input of the function, from previous hidden layer.
            y: Labels.
        """

        return to_variable(-(y * np.log(self.sigmoid_function(X))), "BCELogitLoss")

    def sigmoid_function(self, x: NDArray) -> NDArray:
        """
        Sigmoid Function.
        x: Variable. Derivied from previous hidden layer.
        Return: NDArray
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, x: Variable, y: NDArray):
        return
