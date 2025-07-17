import numpy as np
from numpy.typing import NDArray


class Variable:

    def __init__(self, value: NDArray, label="", children=()):
        self.value = value
        self.label = label
        self._backward = lambda: None
        self.children = children
        self.grad = 0.0

    def __repr__(self) -> str:
        return self.label if self.label else f"Variable: {self.value}"
