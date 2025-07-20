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
        optimizer: Optimizer,
        **kwargs
    ):
        # Data
        self._X = None
        self._y = None

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
        self.optim = optimizer

    @property
    def X(self) -> np.ndarray:
        if self._X is None:
            raise ValueError("The value for X must be set")

        return self._X

    @X.setter
    def X(self, new_X):
        if not isinstance(new_X, np.ndarray):
            raise ValueError("X value must be set with type NDArray")

        self._X = new_X

    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            raise ValueError("The value for y must be set")

        return self._y

    @y.setter
    def y(self, new_y):
        if not isinstance(new_y, np.ndarray):
            raise ValueError("y value must be set with type NDArray")

        self._y = new_y

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

    @property
    def parameters(self):
        params = []

        for i in self.layers_stack.layers:
            if isinstance(i, LinearLayer):
                params.append(i.weights)
                params.append(i.bias)

        return params

    def forward(self) -> Param:
        """
        Defines the forward pass for the neural network model.
        """

        if not isinstance(self.layers_stack, LayerStack):
            logger.error("layers must be defined before model can be trained")

        # Get layers
        layers: tuple[Layer, ...] = self.layers_stack.layers

        # Initialize First Layer
        first_layer = layers[0]
        first_layer.inp = Param(self.X, "inp")

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

    def loss(self) -> np.ndarray:
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
        return self.loss_func.forward(self.y)

    def backward(self):
        """
        Define the backward pass for the neural network
        """

        # Initialize parent layer
        current_layer = self.loss_func
        current_layer.backward(self.y)

        parent_layer = current_layer.parent_layer

        # Continue the backward pass.
        while parent_layer is not None:
            parent_layer.backward()
            parent_layer.parent_layer

    def step(self):
        self.optim.model_parameters = self.parameters
        self.optim.step()

    def zero_grad(self):
        self.optim.model_parameters = self.parameters
        self.optim.zero_grad()
