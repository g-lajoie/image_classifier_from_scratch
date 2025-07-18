import logging
from types import NoneType
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable
from image_classifier.functions.activiation import RELU
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.layers import LayerStack, LinearLayer
from image_classifier.layers.base_layers import Layers

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Neural Network Model.

    Attributes
        data NDArray: Input data for neural network
        layers Sequence[Layers]: Layers for neural network. Layers will be evaluated in sequential order.
    """

    def __init__(self, data: NDArray | None = None, *layers: Layers | None, **kwargs):
        self._data = data

        if (
            isinstance(layers, tuple)
            and isinstance(layers[0], Layers)
            and not isinstance(layers, NoneType)
        ):
            layers = cast(tuple[Layers], layers)
            self._layers_stack = LayerStack(*layers)

    @property
    def data(self):
        """
        Train, Validation, or Test dataset, loaded externally.
        """
        return self._data

    @data.setter
    def data(self, new_data_value):
        """
        Setter function for data property.
        """
        if isinstance(new_data_value, np.ndarray):
            self._data = new_data_value

        else:
            logger.error(
                "The data value must type<NDArray>, got %s",
                new_data_value,
                exc_info=True,
            )

    @property
    def layers_stack(self):
        return self._layers_stack

    @layers_stack.setter
    def layers_stack(self, new_layers_value: LayerStack):
        if not isinstance(new_layers_value, LayerStack):
            logger.error(
                "Layers attribute type must be <LayersStack>, got %s",
                new_layers_value,
                exc_info=True,
            )
            raise

        self._layers_stack = new_layers_value

    def forward(self):
        """
        Defines the forward pass for the neural network model.
        """

        if not isinstance(self.layers_stack, LayerStack):
            logger.error("Layers attribute must be before model can be trained")

        # Initialize First Layer
        first_layer = self.layers_stack.layers[0]
        first_layer.data = self.data
