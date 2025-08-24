import numpy as np


class DataLoader:
    def __init__(self, features: np.ndarray, labels: np.ndarray, batch_size: int):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

        self.batched_data = []

        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("The length of rows for features does not match labels")

    def batch_data(self):
        """
        Batch data for training.
        """

        for i in range(0, self.features.shape[0], self.batch_size):
            X_batch = self.features[i : i + self.batch_size]
            y_batch = self.labels[i : i + self.batch_size]

            self.batched_data.append((X_batch, y_batch))

    def __len__(self):
        return len(self.batched_data)
