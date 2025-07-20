from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from image_classifier.common import Params
from image_classifier.layers.base_layers import Layer


class LossFunction(Layer):
    """
    Interface for loss functions.
    """

    @abstractmethod
    def forward(self, y_true: NDArray, *args, **kwargs):
        """
        Abstract method, overwritten for loss functions.
        """

        raise NotImplementedError(
            "Forward method for loss function has not been implemented."
        )

    @abstractmethod
    def backward(self, y_true: NDArray, *args, **kwargs):
        """
        Abstract method, overwritten for the loss function
        """

        raise NotImplementedError(
            "Backward method for the loss function has not been implemented"
        )
