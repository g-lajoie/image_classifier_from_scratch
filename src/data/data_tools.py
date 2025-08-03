import numpy as np
from numpy.typing import NDArray


def train_test_split(
    data: NDArray, train_amount: float = 0.8, test_amount: float = 0.2
) -> tuple[NDArray, NDArray]:
    """
    Train, test split to split the data into X_train, X_test, y_train, y_test
    """

    if train_amount:
        _test_amount = test_amount

    elif train_amount:
        _test_amount = 1 - train_amount

    else:
        _test_amount = 0.0

    _test_amount = int(_test_amount)

    train, test = data[:_test_amount, :], data[_test_amount + 1 :, :]
    return train, test


def batch_data(data: NDArray, batch_size: int):
    """
    Batch data for training.
    """

    return [data[i : i + 1] for i in range(0, len(data), batch_size)]
