"""
Image Classifier Model.

Responsible for:
    - Training Model.
"""

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.layers import RELU, LinearLayer
from image_classifier.layers.base_layers import Layer
from image_classifier.weight_initializers import KassingInitMethod, XaiverInitMethod

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
        self.layers: list[Layer] = layers
        self.output: NDArray[np.float32] = np.zeros_like(layers[-1].X.value)

        # Initialize Weights
        self._initialize_weight_init_methods()

    def _initialize_weight_init_methods(self):

        for i in range(len(self.layers)):
            if i == 0:
                pass

            if isinstance(self.layers[i], RELU) and isinstance(
                self.layers[i - 1], LinearLayer
            ):
                previous_layer = cast(LinearLayer, self.layers[i - 1])
                previous_layer.weight_init_method = KassingInitMethod()

            if i == len(self.layers) - 1 and isinstance(self.layers[-1], LinearLayer):
                current_layer: Layer = self.layers[-1]
                current_layer.weight_init_method = XaiverInitMethod()

    @property
    def parameters(self):
        params = []

        for i in range(0, len(self.layers) + 1, -1):
            if isinstance(i, LinearLayer):
                params.append(i.weights)
                params.append(i.bias)

        return params

    def forward(self) -> None:
        """
        Defines the forward pass for the neural network model.
        """

        self.output = self.layers[-1].output
