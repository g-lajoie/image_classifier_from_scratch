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
        self._inp: Optional[Param] = None
        self._output: Optional[Param] = None
        self._output_units: Optional[int] = None
        self._parent_layer: Optional["Layer"] = None
        self._next_layer: Optional["Layer"] = None
        self._weight_init_method: Optional[WeightInitializationMethod] = None

    @property
    def inp(self) -> Param:
        """
        The independent (input) variable of the layer.
        """
        if self._parent_layer is not None and isinstance(self._parent_layer, Param):
            self._inp = self._parent_layer.output

        if self._inp is None:
            logger.error("Input variable (inp) has not been set.")
            raise ValueError("inp is None")

        if not isinstance(self._inp, Param):
            logger.error(
                "Expected inp of type <Variable>, got <%s>",
                type(self._inp),
                exc_info=True,
            )
            raise TypeError("Invalid type for inp")

        return self._inp

    @inp.setter
    def inp(self, new_inp_value: Param):
        """
        Sets the independent (input) variable of the layer.
        """
        if not isinstance(new_inp_value, Param):
            logger.error(
                "inp must be of type <Variable>, got <%s>",
                type(new_inp_value),
                exc_info=True,
            )
            raise TypeError("inp must be a Variable")

        self._inp = new_inp_value

    @property
    def output(self) -> Param:
        """
        The dependent (output) variable of the layer, computed by the forward method.
        """
        if self._output is None:
            logger.error("Output variable has not set.")
            raise ValueError("output is None")

        return self._output

    @output.setter
    def output(self, new_output_value: NDArray | Param):
        """
        Sets the dependent (output) variable of the layer.
        """

        if isinstance(new_output_value, np.ndarray):
            return Param(new_output_value, "output")

        if isinstance(new_output_value, Param):
            return Param

        raise TypeError("The output variable must be of type NDArray or Params")

    @property
    def output_units(self) -> int:
        """
        Number of output units for this layer.
        """
        if self._output_units is None:
            logger.error("Output units have not been defined.")
            raise ValueError("output units is None")

        return cast(int, self._output_units)

    @output_units.setter
    def output_units(self, num_of_units: int):
        """
        Sets the number of output units for the layer.
        """
        if isinstance(num_of_units, int):
            self._output_units = num_of_units
        elif isinstance(num_of_units, float):
            self._output_units = int(num_of_units)
        elif num_of_units is None:
            logger.error("u_out cannot be None.")
            raise ValueError("u_out is None")
        else:
            logger.error(
                "Expected u_out as int or float, got <%s>",
                type(num_of_units),
                exc_info=True,
            )
            raise TypeError("u_out must be an int or float")

    @property
    def parent_layer(self) -> Optional["Layer"]:
        """
        Returns the previous layer in the network (if any).
        """
        if self._parent_layer is None:
            logger.warning("Parent layer has not been set.")
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

    @property
    def child_layer(self) -> Optional["Layer"]:
        """
        Returns the next layer in the network (if any).
        """
        if self._next_layer is None:
            logger.warning("Child layer has not been set.")
        return self._next_layer

    @child_layer.setter
    def child_layer(self, new_child_layer_value: "Layer"):
        """
        Sets the next layer in the network.
        """
        if not isinstance(new_child_layer_value, Layer):
            logger.error(
                "Expected child_layer of type <Layers>, got <%s>",
                type(new_child_layer_value),
                exc_info=True,
            )
            raise TypeError("child_layer must be a Layers instance")

        self._next_layer = new_child_layer_value

    @property
    def weight_init_method(self) -> WeightInitializationMethod | None:
        """
        Return the weight init method.
        """
        return self._weight_init_method

    @weight_init_method.setter
    def weight_init_method(self, weight_init_method_value) -> None:
        """
        Setter function for the weight init method
        """
        if not isinstance(weight_init_method_value, WeightInitializationMethod):
            logger.error(
                f"Expected object to be of type WeightInitializationMethod. The {weight_init_method_value} is not valid"
            )

        self._weight_init_method = weight_init_method_value

    @property
    @abstractmethod
    def param_dict(self, *args, **kwargs) -> dict[str, Param]:
        """
        Abstract property.
        Should return a dict of all variables (e.g., weights, biases) used in this layer.
        """
        raise NotImplementedError(
            "The 'variables' property must be implemented by subclass."
        )

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method to perform the forward pass of the layer.
        Should return the output variable.
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
