from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Variable:

    # Define Variables
    value: NDArray = field(repr=False)
    label: str
    backward: Callable = field(default_factory=lambda: lambda: None)
    children: Optional[Iterable] = field(default_factory=list)
    grad: float = 0.0

    # Calculated Variables
    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    # Operations
    def __add__(self, other) -> NDArray:
        """
        Returns a NDArray from the addition of two Variables

        Arguements:
            other: Variable

        Return: NDArray
        """

        if not isinstance(other, Variable):
            raise TypeError(
                f"{other.__name__} is not correct type. Expected Variable, got <{type(other)}>"
            )

        return self.value + other.value

    def __mul__(self, scalar: float) -> NDArray:
        """
        Return a NDArray from the multiplication of two variables

        Arguements:
            other: Variable

        Return: NDArray
        """

        if not isinstance(scalar, float):
            raise TypeError(f"Exepcted type<float>, got <{type(scalar)}>")

        return self.value * scalar

    def __array__(self, dtype=None):
        return np.asarray(self, dtype)
