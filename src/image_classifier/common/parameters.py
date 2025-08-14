import logging
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Param:

    def __init__(self, value, label, grad: NDArray | None = None):
        # Define Variables
        self.value: NDArray = value
        self.label: str = label
        self.grad: NDArray = (
            np.zeros(self.value.shape, dtype=np.float32) if grad is None else grad
        )

    # Calculated Variables
    @property
    def shape(self) -> tuple[int, ...]:
        if self.value is None:
            raise ValueError("Param value is None")

        if not isinstance(self.value, np.ndarray):
            raise TypeError(
                f"Incorrect type for value, expected NDArray, got {type(self.value).__name__}"
            )

        return self.value.shape
