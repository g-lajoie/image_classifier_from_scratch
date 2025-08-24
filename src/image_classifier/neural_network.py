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

    def __init__(self):
        # Layers
        self.l1 = LinearLayer(
            KassingInitMethod(),
            in_features=784,
            out_features=256,
            label="l1",
        )
        self.r1 = RELU(label="r1")
        self.l2 = LinearLayer(
            XaiverInitMethod(),
            in_features=256,
            out_features=62,
            label="l2",
        )

    @property
    def parameters(self):
        """
        Return the weights and bias that are associated with the neural network
        """

        params = []

        for i in [self.l1, self.l2]:
            if isinstance(i, LinearLayer):
                params.append(i.weights)
                params.append(i.bias)

        return params

    def forward(self, batch: np.ndarray):
        """
        Forward pass of the neural network.
        Returns the logits for the last layer of the neural network

        Arguments:
            batch: A numpy array that represents a batch of data.
        """

        l1_logits = self.l1.forward(batch)
        r1_logits = self.r1.forward(l1_logits)
        l2_logits = self.l2.forward(r1_logits)

        return l2_logits

    def backward(self, loss_func_grad):
        """
        Calculates the gradients for the neural networks.

        Agrugments:
            loss_func_grad: A numpy array of the loss function gradients.
        """

        l2_grad = self.l2.backward(loss_func_grad)
        r1_grad = self.r1.backward(l2_grad)
        _ = self.l1.backward(r1_grad)
