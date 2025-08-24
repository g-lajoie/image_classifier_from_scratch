import logging
from abc import ABC, abstractmethod
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common import Param
from image_classifier.weight_initializers.base_weight_initialization import (
    WeightInitializationMethod,
)

logger = logging.getLogger(__name__)


class Layer(ABC):
    """
    Abstract base class for all neural network layers.

    Should be inherited by both linear layers and activation functions.
    Enforces implementation of forward and backward methods,
    and standardizes common interface for handling input/output variables and graph connectivity.
    """

    def __init__(self):
        self._label = None
        self._X = None
        self._in_features = None
        self._out_features = None
        self._parent_layer = None

    @property
    def label(self) -> str | None:
        """
        The label of the layer
        """

        return self._label

    @label.setter
    def label(self, new_label: str) -> None:
        """
        The setter function for the label
        """

        self._label = new_label

    @property
    def X(self) -> Param:
        """
        The independent (input) variable of the layer.
        """

        if not isinstance(self._X, (np.ndarray, Param)):
            raise TypeError(
                f"Expected input variable (X) of type <NDArray or Param>, got <{type(self._X).__name__}>"
            )

        return self._X

    @X.setter
    def X(self, new_X: Param):
        """
        Sets the independent (input) variable of the layer.
        """

        if not isinstance(new_X, (np.ndarray, Param)):
            raise TypeError(
                f"Input variable (X) must be of type <NDArray or Param>, got <{type(new_X).__name__}>"
            )

        self._X = new_X

    @property
    def parent_layer(self) -> Optional["Layer"]:
        """
        Returns the previous layer in the network (if any).
        """

        return self._parent_layer

    @parent_layer.setter
    def parent_layer(self, new_parent_layer_value: "Layer"):
        """
        Sets the previous layer in the network.
        """
        if not isinstance(new_parent_layer_value, Layer):
            logger.error(
                "Expected parent_layer of type <Layers>, got <%s>",
                type(new_parent_layer_value),
                exc_info=True,
            )
            raise TypeError("parent_layer must be a Layers instance")

        self._parent_layer = new_parent_layer_value

    @abstractmethod
    def forward(self, *args, **kwargs) -> NDArray:
        """
        Abstract method to perform the forward pass of the layer.
        Should assign the value of the calculation to self.output
        """
        raise NotImplementedError(
            "The 'forward' method must be implemented by subclass."
        )

    @abstractmethod
    def backward(self, *args, **Kwargs):
        """
        Abstract method to perform the backward pass (gradient calculation).
        """
        raise NotImplementedError(
            "The 'backward' method must be implemented by subclass."
        )
