import logging
from abc import ABC, abstractmethod
from typing import Optional, cast

from numpy.typing import NDArray

from image_classifier.common import Variable

logger = logging.getLogger(__name__)


class Layers(ABC):
    """
    Abstract base class for all neural network layers.

    Should be inherited by both linear layers and activation functions.
    Enforces implementation of forward and backward methods,
    and standardizes common interface for handling input/output variables and graph connectivity.
    """

    def __init__(self):
        self._ind_vars: Optional[Variable] = None
        self._u_out: Optional[int] = None
        self._parent_layer: Optional["Layers"] = None
        self._next_layer: Optional["Layers"] = None

    @property
    def ind_var(self) -> Variable:
        """
        The independent (input) variable of the layer.
        """
        if self._parent_layer is not None and isinstance(self._parent_layer, Variable):
            self._ind_vars = self._parent_layer

        if self._ind_vars is None:
            logger.error("Input variable (ind_var) has not been set.")
            raise ValueError("ind_var is None")

        if not isinstance(self._ind_vars, Variable):
            logger.error(
                "Expected ind_var of type <Variable>, got <%s>",
                type(self._ind_vars),
                exc_info=True,
            )
            raise TypeError("Invalid type for ind_var")

        return self._ind_vars

    @ind_var.setter
    def ind_var(self, new_ind_var: Variable):
        """
        Sets the independent (input) variable of the layer.
        """
        if not isinstance(new_ind_var, Variable):
            logger.error(
                "ind_var must be of type <Variable>, got <%s>",
                type(new_ind_var),
                exc_info=True,
            )
            raise TypeError("ind_var must be a Variable")

        self._ind_vars = new_ind_var

    @property
    def dep_var(self) -> Variable:
        """
        The dependent (output) variable of the layer, computed by the forward method.
        """
        return Variable(self.forward(), "dep_var")

    @property
    def u_out(self) -> int:
        """
        Number of output units for this layer.
        """
        if self._u_out is None:
            logger.error("Output units (u_out) have not been defined.")
            raise ValueError("u_out is None")

        return cast(int, self._u_out)

    @u_out.setter
    def u_out(self, num_of_units: int):
        """
        Sets the number of output units for the layer.
        """
        if isinstance(num_of_units, int):
            self._u_out = num_of_units
        elif isinstance(num_of_units, float):
            self._u_out = int(num_of_units)
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
    def parent_layer(self) -> Optional["Layers"]:
        """
        Returns the previous layer in the network (if any).
        """
        if self._parent_layer is None:
            logger.warning("Parent layer has not been set.")
        return self._parent_layer

    @parent_layer.setter
    def parent_layer(self, new_parent_layer_value: "Layers"):
        """
        Sets the previous layer in the network.
        """
        if not isinstance(new_parent_layer_value, Layers):
            logger.error(
                "Expected parent_layer of type <Layers>, got <%s>",
                type(new_parent_layer_value),
                exc_info=True,
            )
            raise TypeError("parent_layer must be a Layers instance")

        self._parent_layer = new_parent_layer_value

    @property
    def child_layer(self) -> Optional["Layers"]:
        """
        Returns the next layer in the network (if any).
        """
        if self._next_layer is None:
            logger.warning("Child layer has not been set.")
        return self._next_layer

    @child_layer.setter
    def child_layer(self, new_child_layer_value: "Layers"):
        """
        Sets the next layer in the network.
        """
        if not isinstance(new_child_layer_value, Layers):
            logger.error(
                "Expected child_layer of type <Layers>, got <%s>",
                type(new_child_layer_value),
                exc_info=True,
            )
            raise TypeError("child_layer must be a Layers instance")

        self._next_layer = new_child_layer_value

    @property
    @abstractmethod
    def variables(self) -> list[Variable]:
        """
        Abstract property.
        Should return a list of all variables (e.g., weights, biases) used in this layer.
        """
        raise NotImplementedError(
            "The 'variables' property must be implemented by subclass."
        )

    @abstractmethod
    def forward(self) -> NDArray:
        """
        Abstract method to perform the forward pass of the layer.
        Should return the output variable.
        """
        raise NotImplementedError(
            "The 'forward' method must be implemented by subclass."
        )

    @abstractmethod
    def backward(self):
        """
        Abstract method to perform the backward pass (gradient calculation).
        """
        raise NotImplementedError(
            "The 'backward' method must be implemented by subclass."
        )
