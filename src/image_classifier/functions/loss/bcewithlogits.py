import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Params
from image_classifier.layers.base_layers import Layer

from .base_loss_function import LossFunction

logger = logging.getLogger(__name__)


class BCEWithLogits(Layer):
    """
    Binary Cross Entropy with Logit Loss
    """

    def param_dict(self) -> dict[str, Params]:
        """
        Dictionary of all variables in this layer.
        """

        return {"ind_var": self.ind_var}

    def forward(self, y_true: NDArray):
        """
        Binary Cross Entropy with Logit Loss

        Arguments
            x: Input of the function, from previous hidden layer.
            y: Labels.
        """

        return (-(y_true * np.log(self.sigmoid_function(self.ind_var))), "BCELogitLoss")

    def sigmoid_function(self, X: Params) -> NDArray:
        """
        Sigmoid Function.
        x: Variable. Derivied from previous hidden layer.
        Return: NDArray
        """
        if self.ind_var.value is None:
            logger.error("The value for %s cannot be none", self.ind_var.label)
            raise ValueError(f"Value for {self.ind_var.label} is none")

        return 1 / (1 + np.exp(-self.ind_var.value))

    def backward(self, x: Params, y: NDArray):
        return
