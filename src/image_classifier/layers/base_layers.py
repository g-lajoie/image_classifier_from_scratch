from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Layers(ABC):
    """
    Base Class for all Neural Network Layers. Should be inherited by both Linear Layers and Activaition functions.
    Includes abstract methods for forward used in model training and inference, and backward used in backpropagation.
    """

    @property
    @abstractmethod
    def data(self) -> NDArray | None:
        """
        Abstract method for the data property
        """

        raise NotImplementedError("The data attribute has not been implemented")

    @data.setter
    @abstractmethod
    def data(self, new_data_value) -> NDArray | None:
        """
        Setter for the data property.
        """

        raise NotImplementedError(
            "The setter for the data property has not been impletemented"
        )

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
