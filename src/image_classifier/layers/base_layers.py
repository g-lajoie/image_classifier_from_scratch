from abc import ABC, abstractmethod

from numpy.typing import NDArray

from image_classifier.common import Variable


class Layers(ABC):
    """
    Base Class for all Neural Network Layers. Should be inherited by both Linear Layers and Activaition functions.
    Includes abstract methods for forward used in model training and inference, and backward used in backpropagation.
    """

    @property
    @abstractmethod
    def ind_vars(self) -> Variable:
        """
        Abstract method for the independent variable.
        """

        raise NotImplementedError("The ind_vars property has not been implemented")

    @ind_vars.setter
    @abstractmethod
    def ind_vars(self, data) -> Variable:
        """
        Setter for the independent variable property
        """

        raise NotImplementedError(
            "The setter method for the ind_vars property has not been impletemented"
        )

    @property
    @abstractmethod
    def dep_vars(self) -> Variable:
        """
        Abstract method for dependent variable property.
        """

        raise NotImplementedError("The dep_vars method has not been implemented")

    @dep_vars.setter
    def dep_vars(self, new_dep_var) -> Variable:
        """
        Setter for the dependent variable property
        """

        raise NotImplementedError(
            "The setter method for the dep_vars property has not been implemented"
        )

    @property
    def variables(self):
        """
        Abstaract method for variables.
        List of all the variables for the layer
        """

        raise NotImplementedError("The variables list has not been initialized")

    @abstractmethod
    def forward(self):
        """
        Abstract method for forward pass.
        """

        raise NotImplementedError("The forward method has not been implemented")

    @abstractmethod
    def backward(self):
        """
        Abstract method for backward pass
        """

        raise NotImplementedError("The backward method has not been implemented")
