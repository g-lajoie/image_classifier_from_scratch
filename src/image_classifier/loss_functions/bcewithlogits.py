import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.layers.base_layers import Layer

from .base_loss_function import LossFunction

logger = logging.getLogger(__name__)


class BCEWithLogits(LossFunction):
    """
    Binary Cross Entropy with Logit Loss
    """

    def __init__(self, inp: Param):
        self.input = inp

    @property
    def param_dict(self) -> dict[str, Param]:
        """
        Dictionary of all variables in this layer.
        """

        return {"ind_var": self.input}

    def calculate(self, y_true: NDArray) -> np.ndarray:
        """
        Binary Cross Entropy with Logit Loss

        Arguments
            x: inp of the function, from previous hidden layer.
            y: Labels.
        """

        return -(y_true * np.log(self.sigmoid_function(self.input)))

    def sigmoid_function(self, X: Param) -> NDArray:
        """
        Sigmoid Function.
        x: Variable. Derivied from previous hidden layer.
        Return: NDArray
        """
        if self.input.value is None:
            logger.error("The value for %s cannot be none", self.input.label)
            raise ValueError(f"Value for {self.input.label} is none")

        return 1 / (1 + np.exp(-self.input.value))

    def backward(self, x: Param, y: NDArray) -> np.ndarray:
        return np.zeros_like(0)
