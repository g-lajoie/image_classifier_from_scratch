import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class Variable:

    # Define Variables
    value: Optional[NDArray] = field(repr=False)
    label: str
    backward: Callable = field(default_factory=lambda: lambda: None)
    children: Optional[Iterable] = field(default_factory=list)
    grad: float = 0.0

    # Calculated Variables
    @property
    def shape(self) -> tuple[int, ...]:
        if self.value is None:
            logger.error("Could not get value for variable, value not defined")

        if not isinstance(self.value, np.ndarray):
            logger.error(
                "Incorrect type for value, expected NDArray, got %s",
                type(self.value),
                exc_info=True,
            )

        value = cast(NDArray, self.value)
        return value.shape

    # Operations
    def __add__(self, other) -> NDArray:
        """
        Returns a NDArray from the addition of two Variables

        Arguements:
            other: Variable

        Return: NDArray
        """
        if self.value is None:
            logger.error("Could not get value for variable, value not defined")

        if not isinstance(self.value, np.ndarray):
            logger.error(
                "Incorrect type for value, expected NDArray, got %s",
                type(self.value),
                exc_info=True,
            )

        if not isinstance(other, Variable):
            raise TypeError(
                f"{other.__name__} is not correct type. Expected Variable, got <{type(other)}>"
            )

        value = cast(NDArray, self.value)

        return value + other.value

    def __mul__(self, scalar: float) -> NDArray:
        """
        Return a NDArray from the multiplication of two variables

        Arguements:
            other: Variable

        Return: NDArray
        """
        if self.value is None:
            logger.error("Could not get value for variable, value not defined")

        if not isinstance(self.value, np.ndarray):
            logger.error(
                "Incorrect type for value, expected NDArray, got %s",
                type(self.value),
                exc_info=True,
            )

        if not isinstance(scalar, float):
            raise TypeError(f"Exepcted type<float>, got <{type(scalar)}>")

        value = cast(NDArray, self.value)

        return value * scalar

    def __array__(self, dtype=None):
        if self.value is None:
            logger.error("Could not get value for variable, value not defined")

        if not isinstance(self.value, np.ndarray):
            logger.error(
                "Incorrect type for value, expected NDArray, got %s",
                type(self.value),
                exc_info=True,
            )

        return np.asarray(self, dtype)
