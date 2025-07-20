"""
Image Classifier Model.

Responsible for:
    - Training Model.
"""

import logging
from types import NoneType
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray

from data import DataLoader
from image_classifier.common.parameters import Param
from image_classifier.functions.activiation import RELU
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.functions.loss.base_loss_function import LossFunction
from image_classifier.layers import LayerStack, LinearLayer
from image_classifier.layers.base_layers import Layer
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
        *layers: Layer | LayerStack,
        loss_func: LossFunction,
        optim: Optimizer,
        **kwargs
    ):
        # LayerStack
        if isinstance(layers, tuple) and all(
            [isinstance(layer, Layer) for layer in layers]
        ):
            layers = cast(tuple[Layer], layers)
            self._layers_stack = LayerStack(*layers)

        elif isinstance(layers, tuple) and all(
            [isinstance(layer, LayerStack) for layer in layers]
        ):
            combined_layers = []

            for layer in layers:
                layer = cast(LayerStack, layer)
                combined_layers.append(layer.layers)

            self._layers_stack = LayerStack(*combined_layers)

        # Loss Function
        if not isinstance(loss_func, LossFunction):
            raise TypeError(
                "The loss function must be type of or subclass of <LossFunction>"
            )

        self.loss_func = loss_func

        # Optimizer
        if not isinstance(optim, Optimizer):
            raise TypeError(
                "The optimizer must be type of or subtclasso of <Optimizer>"
            )

        self.optim = optim

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

    def forward(self, X: NDArray) -> Param:
        """
        Defines the forward pass for the neural network model.
        """

        if not isinstance(self.layers_stack, LayerStack):
            logger.error("layers must be defined before model can be trained")

        # Get layers
        layers: tuple[Layer, ...] = self.layers_stack.layers

        # Initialize First Layer
        first_layer = layers[0]
        first_layer.inp = Param(X, "inp")

        # Start foward pass
        for i in range(1, len(layers)):
            layers[i].forward()

        # Return Regression Output or Logits
        last_layer = self.layers_stack.layers[-1]

        if not isinstance(last_layer, LinearLayer):
            logger.error("The last layers of the model should be a linear layers")
            raise ValueError("The last layers of the model is not a Linear Layer")

        if last_layer.output is None:
            raise ValueError("The output of the model is none")

        self.out_put = last_layer.output
        return self.out_put

    def loss(self, y) -> np.ndarray:
        """
        Calculate the provided loss function.
        """

        try:
            out_put = self.out_put
        except AttributeError:
            logger.error("Could not find output attribute, please run forward pass")
            raise

        self.loss_func.inp = out_put
        self.loss_func.parent_layer = self.layers_stack.layers[0]
        return self.loss_func.forward(y)

    def backward(self, y):
        """
        Define the backward pass for the neural network
        """

        # Initialize parent layer
        current_layer = self.loss_func
        current_layer.backward(y)

        parent_layer = current_layer.parent_layer

        # Continue the backward pass.
        while parent_layer is not None:
            parent_layer.backward()
            parent_layer.parent_layer

    def optimizer_step(self):
        """
        Update the parameters.
        """

        pass

    def zero_grad(self):
        """
        Zero out the gradient.
        """

        pass
