"""
Image Classifier Model.

Responsible for:
    - Training Model.
"""

import logging
from typing import Iterable, cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.layers import LinearLayer
from image_classifier.layers.base_layers import Layer
from image_classifier.loss_functions.base_loss_function import LossFunction
from image_classifier.optimizer.base_optimizer import Optimizer

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Neural Network Model.

    Attributes
        data NDArray: Input data for neural network
        layers Sequence[Layers]: Layers for neural network. Layers will be evaluated in sequential order.
    """

    def __init__(
        self,
        layers: list[Layer],
        **kwargs,
    ):
        # Layers
        self.layers = layers

    @property
    def parameters(self):
        params = []

        for i in range(0, len(self.layers) + 1, -1):
            if isinstance(i, LinearLayer):
                params.append(i.weights)
                params.append(i.bias)

        return params

    def forward(self) -> Param:
        """
        Defines the forward pass for the neural network model.
        """

        for layer in self.layers:
            layer.forward()

        return self.layers[-1].output

    def backward(self, loss_func: LossFunction, labels: NDArray):
        """
        Define the backward pass for the neural network
        """

        # Initialize parent layer
        loss_func.backward()

        current_layer = self.layers[-1]

        # Continue the backward pass.
        while current_layer is not None:
            current_layer.backward()
            current_layer = current_layer.parent_layer
