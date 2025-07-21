"""
Smoke test runner - for development only.
Not part of production code.
"""

import logging
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data.data_tools import batch_data, train_test_split
from image_classifier.functions.activiation import RELU
from image_classifier.functions.loss import CatCrossEntropy
from image_classifier.layers import LayerStack, LinearLayer
from image_classifier.layers.weights_initialization import WeightInitMethod
from image_classifier.neural_network import NeuralNetwork
from image_classifier.optimizer import Adam

# Configure Logging.
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def define_layers() -> LayerStack:
    """
    Define Linear Layers, Acitivation Function, and Loss Function.
    """

    logger.info(
        "Defining Linear Layers, Activiation Functions, and Loss Functions for neural network"
    )

    layers = LayerStack(
        LinearLayer(u_out=128, weight_init_method=WeightInitMethod.HE),
        RELU(),
        LinearLayer(u_out=256, weight_init_method=WeightInitMethod.HE),
        RELU(),
        LinearLayer(u_out=10, weight_init_method=WeightInitMethod.XAVIER),
    )

    logger.info("Neural network layers successfully created.")

    return layers


def main() -> None:

    # Load Data
    logger.info("Loading MNIST Dataset")

    FILE_PATH = Path.cwd().parent / "tmp/mnist_raw/emnist-byclass-train.csv"
    data = np.loadtxt("tmp/mnist_raw/emnist-byclass-train.csv", delimiter=",")
    train, val = train_test_split(data)

    logger.info("Complete the load of the dataset")

    # Define Model Structure
    layers = define_layers()  # Layers
    loss_fn = CatCrossEntropy()  # Loss Function
    optim = Adam()

    # Initialize Model
    model = NeuralNetwork(layers=layers, loss_func=loss_fn, optimizer=optim)

    for epoch in range(100):
        for train in batch_data(train, batch_size=64):

            X_train, y_train = train[:1, :], train[1:, :]

            # Load data to the model
            model.X = X_train
            model.y = y_train

            # Zero out the gradient
            optim.zero_grad()

            # Forward pass
            model.forward()

            # Compute the Loss
            loss = model.loss()
            print(loss)

            # Back Propgratmion
            model.backward()

            # Optimize the params.
            model.step()


if __name__ == "__main__":
    main()
