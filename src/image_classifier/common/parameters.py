import logging
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Param:

    def __init__(self, value, label, grad: NDArray[np.float64], shape: tuple[int, ...]):
        # Define Variables
        self._value: NDArray[np.float64] = value
        self.label: str = label
        self._grad: NDArray[np.float64] = grad
        self.shape: tuple[int, ...] = shape

    @property
    def value(self):
        if self._value.shape != self.shape:
            raise ValueError(f"Possible mismatch in shape in {self.label}")

        return self._value

    @value.setter
    def value(self, new_value: NDArray[np.float64]):
        self._value = new_value

    @property
    def grad(self):
        if self._grad.shape != self.shape:
            raise ValueError(f"Possible mistmatch in shape in {self.label}")

        return self._grad

    @grad.setter
    def grad(self, new_grad: NDArray[np.float64]):
        self._grad = new_grad
