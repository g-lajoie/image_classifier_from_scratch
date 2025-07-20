"""
Data Loader for Neural Network.

Responsible for:
    - Ensuring data is loaded completely from source.
    - Handling missing values.
    - Splitting Data
    - Creating Mini Batches
"""

import numpy as np
from numpy.typing import NDArray


class DataLoader:
    """
    Dataloader for the neural network.
    """

    def __init__(self, data: NDArray):
        self.X_train = np.zeros_like(np.array(1))
        self.X_test = np.zeros_like(np.array(1))
        self.y_train = np.zeros_like(np.array(1))
        self.y_test = np.zeros_like(np.array(1))

    @property
    def train_batch(self) -> tuple[NDArray, NDArray]:
        return self.X_train, self.y_train
