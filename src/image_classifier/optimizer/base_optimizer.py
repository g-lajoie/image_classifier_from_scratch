from abc import abstractmethod

from image_classifier.common import Param


class Optimizer:
    """
    Interface for optimizer classes.
    """

    @property
    def model_parameters(self) -> list[Param]:
        if self._model_parameters is None:
            raise ValueError("The model parameter attribute must not be None")

        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, model_params: list[Param]):
        if not isinstance(model_params, list) and not all(
            [True for param in model_params if isinstance(param, Param)]
        ):

            raise ValueError(
                "The type for model params is incorrect. Expected list[Param]"
            )

        self._model_parameters = model_params

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
