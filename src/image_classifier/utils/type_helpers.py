import numpy as np
from numpy.typing import NDArray

from common.variable import Variable


def to_ndarry(input: Variable | NDArray) -> NDArray:
    """
    Converts class of type Variable or NDAarry to NDArray
    """

    if isinstance(input, Variable):
        return input.value

    if isinstance(input, np.ndarray):
        return input

    raise TypeError(f"Expected object of type Variable or NDArray, got{type(input)}")


def to_variable(input: Variable | NDArray, label: str) -> Variable:
    """
    Ensures that Variable is returned
    """

    if isinstance(input, Variable):
        return input

    if isinstance(input, np.ndarray):
        return Variable(input, label=label)
