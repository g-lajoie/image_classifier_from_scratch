import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Params

from .base_loss_function import LossFunction

logger = logging.getLogger(__name__)


class BCEWithLogits(LossFunction, Layer):
    """
    Binary Cross Entropy with Logit Loss
    """

    def forward(self, X: Params, y: NDArray):
        """
        Binary Cross Entropy with Logit Loss

        Arguments
            x: Input of the function, from previous hidden layer.
            y: Labels.
        """

        return (-(y * np.log(self.sigmoid_function(X))), "BCELogitLoss")

    def sigmoid_function(self, X: Params) -> NDArray:
        """
        Sigmoid Function.
        x: Variable. Derivied from previous hidden layer.
        Return: NDArray
        """
        if X.value is None:
            logger.error("The value for %s cannot be none", X.label)
            raise ValueError(f"Value for {X.label} is none")

        return 1 / (1 + np.exp(-X.value))

    def backward(self, x: Params, y: NDArray):
        return
