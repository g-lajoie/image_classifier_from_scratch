from abc import ABC, abstractmethod


class Layers(ABC):
    """
    Base Class for all Neural Network Layers. Should be inherited by both Linear Layers and Activaition functions.
    Includes abstract methods for forward used in model training and inference, and backward used in backpropagation.
    """

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
