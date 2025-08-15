from abc import abstractmethod

from image_classifier.common import Param


class Optimizer:
    """
    Interface for optimizer classes.
    """

    @abstractmethod
    def step(self, *args, **kwargs):
        """
        Method to update the parameters
        """

        raise NotImplementedError(
            "The update parameters method has not been implemented"
        )

    @abstractmethod
    def zero_grad(self, *args, **kwargs):
        """
        Method to zero gradients
        """

        raise NotImplementedError("The zero_grad method has not been implemented")
