from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from image_classifier.common import Param
from image_classifier.layers import LinearLayer

from .base_optimizer import Optimizer


class Adam(Optimizer):
    """
    The ADAM optimizer.
    """

    def __init__(self):

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.alpha: float = 0.01
        self.epsilon: float = 1e-8
        self.t = 0

    @property
    def model_parameters(self) -> list[Param]:
        """
        Property that defines the model parameteres
        """
        if self._model_parameters is None:
            pass

        return self._model_parameters

    @model_parameters.setter
    def model_parmaeters(self, linear_layers: list[LinearLayer]):
        """
        Helper function that assigns the parameters to the optimizer
        """

        parameters: list[Param | None] = [
            value for layer in linear_layers for value in layer.param_dict.values()
        ]

        if not all(isinstance(p, Param) for p in parameters):
            raise ValueError("All model parameters must be non-None and of type Param")

        self._model_parameters: list[Param] = [
            p for p in parameters if isinstance(p, Param)
        ]

    def step(self):

        self.m = {param: np.zeros_like(param.value) for param in self.model_parameters}
        self.v = {param: np.zeros_like(param.value) for param in self.model_parameters}
        print(self.m)
        print(self.v)

        self.t += 1
        for param in self.model_parameters:
            g = param.grad

            self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * g
            self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * (g**2)

            # Bias correction
            m_hat = self.m[param] / (1 - self.beta_1**self.t)
            v_hat = self.v[param] / (1 - self.beta_2**self.t)

            param.value -= self.alpha * (m_hat / (v_hat + self.epsilon))
            print(param.value)

    def zero_grad(self):

        for param in self.model_parameters:

            if param is None:
                raise ValueError(
                    "A parameter was found to be None, Params cannot be None"
                )

            param.grad = np.zeros_like(param.value)
