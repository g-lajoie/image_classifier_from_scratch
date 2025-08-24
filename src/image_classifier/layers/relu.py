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

    def __init__(self, label: str):
        super().__init__()

        self.label = label

    def __repr__(self):
        return f"[{self.label}]: RELU Layer"

    def forward(self, inp: np.ndarray) -> NDArray:
        """
        Caclulates the ReLU function.
        """

        self.X = Param(
            inp, label=f"Relu: {self.label}", grad=np.zeros_like(inp), shape=inp.shape
        )
        return np.maximum(0, self.X.value)

    def backward(self, previous_layer_grad: np.ndarray) -> np.ndarray:
        """
        Calculate the dervative for the RELU funcion.
        """

        if self.X.value is None:
            logger.error("The value for the %s cannot be None.", self.X.label)
            raise ValueError(f"The value for {self.X.label} is none.")

        self.X.grad = (self.X.value > 0).astype(float) * previous_layer_grad
        return self.X.grad
