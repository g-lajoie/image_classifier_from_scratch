import numpy as np
from numpy.typing import NDArray


def train_test_split(
    data: NDArray, train_amount: float = 0.8
) -> tuple[NDArray, NDArray]:
    """
    Train, test split to split the data into X_train, X_test, y_train, y_test
    """

    parition_index = int(data.shape[0] * train_amount)
    return data[:parition_index, :], data[parition_index + 1 :, :]


def batch_data(data: NDArray, batch_size: int):
    """
    Batch data for training.
    """

    return [data[i : i + 1] for i in range(0, len(data), batch_size)]


def one_hot_encoding(data: NDArray, number_of_classes: int) -> np.ndarray:
    """
    One hot encode the data
    """

    if data.size != 1:
        ValueError(
            "The dimension for data is less than 0, or greater than 1, another implementaion of one-hot-encoding is required"
        )

    one_hot = np.zeros((data.size, number_of_classes), dtype=np.float32)
    return one_hot[np.arange(data.size), data]
