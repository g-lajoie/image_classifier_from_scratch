import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.layers import LinearLayer
from image_classifier.layers.base_layers import Layers

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Neural Network Model.

    Attributes
        data NDArray: Data for
    """

    def __init__(self, data: NDArray, layers: Sequence[Layers]):
        self.data = data
        self.layers = layers

    def forward(self):
        """
        Defines the forward pass for the neural network model.
        """

        # Define input layer
        first_layer = self.layers[0]

        if isinstance(first_layer, LinearLayer):
            first_layer.data = self.data

        else:
            logger.error(
                "The first layer must be of type <LinearLayer>, instead got: %s",
                type(first_layer),
                exc_info=True,
            )
            raise

        # Run forward pass through sequential layers
        previous_layer = first_layer

        for i in range(1, len(self.layers)):
            self.layers[i].data = previous_layer.forward()
            previous_layer = self.layers[i]
