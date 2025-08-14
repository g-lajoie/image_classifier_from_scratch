import numpy as np
from numpy.typing import NDArray


def reshape_for_matmul(value: NDArray | None, other: NDArray) -> np.ndarray:
    """
    Preforms data validation for the np matricies and if needed transposes matrices for correct matrix multiplication
    """

    # Data Validation
    if value is None or not isinstance(value, np.ndarray):
        raise TypeError("The value argument is none or incorrect")

    if other is None or not isinstance(other, np.ndarray):
        raise TypeError("The other argument is none or incorrect")

    if value.ndim == 0 or other.ndim == 0:
        return value

    # Matrix Multipication
    if value.shape[1] == other.shape[0]:
        return value @ other

    if value.shape[0] == other.shape[0]:
        return np.transpose(value) @ other

    elif value.shape[1] == other.shape[1]:
        return value @ np.transpose(other)

    elif value.shape[0] == other.shape[1]:
        return np.transpose(value) @ np.transpose(other)

    else:
        raise ValueError("The numpy arrays are incompatible.")
