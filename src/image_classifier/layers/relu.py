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

    def __init__(self, previous_layer: Layer, label: str):
        super().__init__()

        if previous_layer is None:
            raise ValueError("The previous layer for {self.label} is None")

        self.X: Param = Param(
            np.zeros_like(previous_layer.X.value, dtype=np.float32),
            label=f"RELU: {self.label}",
        )
        self.parent_layer = previous_layer
        self.units = previous_layer.units
        self.label = label

    def __repr__(self):
        return f"<self.label>: RELU Layer - Units:{self.units}"

    @property
    def param_dict(self) -> dict[str, Param]:
        """
        List of all the parameters for the layer
        """

        return {"ind_var": self.X}

    def forward(self) -> NDArray:
        """
        Caclulates the ReLU function.
        """

        if not self.parent_layer:
            raise ValueError("Parent layer's forward method has not been computed")

        self.X.value = self.parent_layer.output

        return np.maximum(0, self.X.value)

    def backward(self, previous_layer: Layer):
        """
        Calculate the dervative for the RELU funcion.
        """

        if self.X.value is None:
            logger.error("The value for the %s cannot be None.", self.X.label)
            raise ValueError(f"The value for {self.X.label} is none.")

        self.X.grad = (self.X.value > 0).astype(float) @ previous_layer.X.grad
