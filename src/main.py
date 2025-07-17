import numpy as np
from numpy.typing import NDArray

from common.enums import WeightInitiailizationMethod as WeightInitMethod
from common.variable import Variable
from neural_network import NeuralNetwork
from weights_initialization import ScaledInitializer


def main(input: NDArray):

    # Initialize Variables
    weights_init = ScaledInitializer(WeightInitMethod.XAIVER)

    # Define layers
    nn_1 = NeuralNetwork()

    X = Variable(input, label="X")
    W = Variable(weights_init.init_weights(X, 128))

    # Forward Pass
