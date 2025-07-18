from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray


class VariableType(str, Enum):

    LAYER = "layer"
    ACTIVATION = "activation"
    LOSS = "loss"


@dataclass
class Variable:

    value: NDArray
    label: str
    variable_type: VariableType
    backward: Callable = lambda: None
    children: Optional[Iterable] = None
    grad: float = 0.0
