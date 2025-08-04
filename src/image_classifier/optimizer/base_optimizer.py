from abc import abstractmethod

from image_classifier.common import Param


class Optimizer:
    """
    Interface for optimizer classes.
    """

    def __init__(self):
        self._model_parameters = None

    @property
    def model_parameters(self) -> list[Param]:
        if self._model_parameters is None:
            raise ValueError("The model parameter attribute must not be None")

        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, new_model_params_value: list[Param]):
        if not isinstance(new_model_params_value, list) and not all(
            [True for param in new_model_params_value if isinstance(param, Param)]
        ):

            raise ValueError(
                "The type for model params is incorrect. Expected list[Param]"
            )

        self._model_parameters = new_model_params_value

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
