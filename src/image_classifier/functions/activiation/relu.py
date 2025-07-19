import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Params
from image_classifier.layers.base_layers import Layer

from .base_activation_function import ActivationFunction

logger = logging.getLogger(__name__)


class RELU(ActivationFunction, Layer):
    """
    ReLU: Rectified Linear Unit
    """

    def __init__(self, ind_var=None):
        pass

    @property
    def param_dict(self) -> dict[str, Params]:
        """
        List of all the parameters for the layer
        """

        return {"ind_var": self.ind_var}

    def forward(self):
        """
        Caclulates the ReLU function.
        """
        return np.maximum(0, self.ind_var)

    def backward(self):
        """
        Calculate the dervative for the RELU funcion.
        """
        if self.ind_var.value is None:
            logger.error("The value for the %s cannot be None.", self.ind_var.label)
            raise ValueError(f"The value for {self.ind_var.label} is none.")

        return (self.ind_var.value > 0).astype(float)
