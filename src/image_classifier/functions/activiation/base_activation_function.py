from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Interface for activaition class.
    """

    def function(self):
        raise NotImplementedError("The function method has not been implemented.")
