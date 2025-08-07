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

    def __init__(self, input: Layer):
        super().__init__()

        self.input = input.output

    @property
    def param_dict(self) -> dict[str, Param]:
        """
        List of all the parameters for the layer
        """

        return {"ind_var": self.input}

    def forward(self) -> None:
        """
        Caclulates the ReLU function.
        """

        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self):
        """
        Calculate the dervative for the RELU funcion.
        """

        if self.input.value is None:
            logger.error("The value for the %s cannot be None.", self.input.label)
            raise ValueError(f"The value for {self.input.label} is none.")

        if self.child_layer is None:
            logger.error(
                "The child layer for the RELU layer cannot be None. ReLU is a hidden layer.",
                self.input.label,
            )
            raise ValueError(f"The child layer for ReLU is none.")

        self.input.grad = (self.input.value > 0).astype(
            float
        ) @ self.child_layer.input.grad
