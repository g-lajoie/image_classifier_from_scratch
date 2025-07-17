from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.variable import Variable
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.functions.loss.base_loss_function import LossFunction
from image_classifier.layers import LinearLayer


class NeuralNetwork:
    """
    Neural Network Model.

    Attributes
        data NDArray: Data for
    """

    def __init__(
        self,
        data: NDArray,
        layers: Iterable[tuple[LinearLayer, ActivationFunction | LossFunction]],
    ):
        pass

    def forward(self):
        """
        Defines the forward pass for the neural network model.
        """
        pass
