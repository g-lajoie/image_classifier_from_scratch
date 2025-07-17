"""
Smoke test runner - for development only.
Not part of production code.
"""

import numpy as np
from numpy.typing import NDArray

from common.enums import WeightInitiailizationMethod as WeightInitMethod
from common.variable import Variable
from linear_layer import LinearLayer
from neural_network import NeuralNetowrk
from src.functionas.activation_functions import RELU, BCEWithLogits
from weights_initialization import ScaledInitializer


def main(input: NDArray):

    # Define layers
    nn_1 = LinearLayer(
        weight_init=ScaledInitializer(WeightInitMethod.HE),
        activation_function=RELU(),
        u_out=128,
    )

    nn_2 = LinearLayer(
        weight_init=ScaledInitializer(WeightInitMethod.HE),
        activation_function=RELU(),
        u_out=256,
    )

    nn_3 = LinearLayer(
        weight_init=ScaledInitializer(WeightInitMethod.XAIVER),
        activation_function=BCEWithLogits(),
        u_out=1,
    )

    layers = [nn_1, nn_2, nn_3]

    # Intialize model
    model = NeuralNetowrk(layers)

    # Forward Pass
    model.forward()
