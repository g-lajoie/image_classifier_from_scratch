import numpy as np
from numpy.typing import NDArray


def reshape_for_matmul(value: NDArray | None, other: NDArray) -> bool:

    if value is None or not isinstance(value, np.ndarray):
        raise TypeError("The value argument is none or incorrect")

    if other is None or not isinstance(value, np.ndarray):
        raise TypeError("The other argument is none or incorrect")

    if isinstance(value, np.ndarray):
        raise ValueError("The value argument must be NDArray")

    if isinstance(other, np.ndarray):
        raise ValueError("The other argument must be NDArray")

    if value.ndim == 0 or other.ndim == 0:
        return value

    if value.shape[0] == value.shape[1]:
        return value

    elif value.shape[0] == value.shape[0]:
        return value.T

    else:
        raise ValueError("The numpy arrays are incompatible.")
