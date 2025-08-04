import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.layers.base_layers import Layer

logger = logging.getLogger(__name__)


class RELU(Layer):
    """
    ReLU: Rectified Linear Unit
    """

    def __init__(self, ind_var=None):
        pass

    @property
    def param_dict(self) -> dict[str, Param]:
        """
        List of all the parameters for the layer
        """

        return {"ind_var": self.inp}

    def forward(self):
        """
        Caclulates the ReLU function.
        """
        return np.maximum(0, self.inp)

    def backward(self):
        """
        Calculate the dervative for the RELU funcion.
        """

        if self.inp.value is None:
            logger.error("The value for the %s cannot be None.", self.inp.label)
            raise ValueError(f"The value for {self.inp.label} is none.")

        if self.child_layer is None:
            logger.error(
                "The child layer for the RELU layer cannot be None. ReLU is a hidden layer.",
                self.inp.label,
            )
            raise ValueError(f"The child layer for ReLU is none.")

        self.inp.grad = (self.inp.value > 0).astype(float) @ self.child_layer.inp.grad
