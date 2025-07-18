"""
Smoke test runner - for development only.
Not part of production code.
"""

import logging
import subprocess

import numpy as np
from numpy.typing import NDArray

from image_classifier.common.enums import (
    WeightInitiailizationMethod as WeightInitMethod,
)
from image_classifier.common.variable import Variable
from image_classifier.functions.activiation import RELU
from image_classifier.functions.activiation.base_activation_function import (
    ActivationFunction,
)
from image_classifier.functions.loss import BCEWithLogits
from image_classifier.layers import LinearLayer
from image_classifier.layers.weights_initialization import ScaledInitializer
from neural_network import NeuralNetwork

# Configure Logging.
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def define_layers() -> list[LinearLayer | ActivationFunction]:
    """Define Linear Layers, Acitivation Function, and Loss Function."""
    logger.info(
        "Defining Linear Layers, Activiation Functions, and Loss Functions for neural network"
    )

    nn_1 = LinearLayer(
        weight_init=ScaledInitializer(WeightInitMethod.HE),
        u_out=128,
    )

    nn_2 = LinearLayer(
        weight_init=ScaledInitializer(WeightInitMethod.HE),
        u_out=256,
    )

    nn_3 = LinearLayer(
        weight_init=ScaledInitializer(WeightInitMethod.XAVIER),
        u_out=1,
    )

    relu = RELU()  # Activation Function
    loss_fn = BCEWithLogits()  # Loss Function

    logger.info("Neural network layers successfully created.")

    # Combine all layers
    return [nn_1, relu, nn_2, relu, nn_3]


def get_data() -> None:
    """
    Get MNIST Dataset and ensure it conforms to shape(b, c, n, m)

    b: batch_size
    c: channel = 1, Only dealing with grayscale images.
    n: number of
    m: number of features.
    """
    logger.info("Downloading MNIST Dataset")

    try:
        subprocess.run(["python", "scripts/load_data.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Error when retrieving data...", exc_info=True)
        raise

    with open("data/mnist_raw/train-images-idx3-ubyte.gz", "rb") as file:
        raw_data = file.read()

    data = np.frombuffer(raw_data)
    print(data)


def main() -> None:

    # Load Data
    data = get_data()

    # Define Model
    layers = define_layers()
    model = NeuralNetwork(data, layers)


if __name__ == "__main__":
    main()
