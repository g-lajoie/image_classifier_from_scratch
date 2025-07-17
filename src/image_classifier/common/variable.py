from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class VariableType(str, Enum):

    LAYER = "layer"
    ACTIVATION = "activation"
    LOSS = "loss"


class Variable(BaseModel):

    value: NDArray
    label: str
    variable_type: VariableType
    backward: Callable = lambda: None
    children: Optional[Iterable] = None
    grad: float = 0.0
