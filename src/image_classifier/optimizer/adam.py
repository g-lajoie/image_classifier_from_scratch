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

    def __init__(self, parameters: list[Param]):

        self.model_parameters = parameters

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.alpha: float = 0.001
        self.epsilon: float = 1e-8
        self.t = 0

        self.m = {param: np.zeros_like(param.grad) for param in self.model_parameters}
        self.v = {param: np.zeros_like(param.grad) for param in self.model_parameters}

    def step(self):

        self.t += 1
        for param in self.model_parameters:
            g = param.grad

            self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * g
            self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * (g**2)

            # Bias correction
            m_hat = self.m[param] / (1 - self.beta_1**self.t)
            v_hat = self.v[param] / (1 - self.beta_2**self.t)

            # Update parameters
            param.value -= self.alpha * (m_hat / (np.sqrt(v_hat) + self.epsilon))

    def zero_grad(self):

        for param in self.model_parameters:

            if param is None:
                raise ValueError(
                    "A parameter was found to be None, Params cannot be None"
                )

            param.grad = np.zeros_like(param.value)
