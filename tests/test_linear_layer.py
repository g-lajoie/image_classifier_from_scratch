import numpy as np
from numpy.random import PCG64, Generator
from numpy.typing import NDArray

from src.activation_functions.relu import RELU
from src.linear_layer import LinearLayer
from src.weights_initialization import RandomInitializer

random_generator = Generator(PCG64())


class TestLinearLayer:

    random_init = RandomInitializer()
    activation_fn = RELU()

    def test_linear_layer_output_shape(self):
        """
        est case for the output shape for the Linear Layer
        """

        test_inputs = [
            random_generator.normal(0, 1, size=(3, 2)),
            random_generator.normal(0, 1, size=(10, 4)),
            random_generator.normal(0, 1, size=(50, 12)),
        ]

        for X in test_inputs:

            linear_layer = LinearLayer(
                weight_init=self.random_init,
                activation_function=self.activation_fn,
                u_out=int(random_generator.uniform(1, 256) // 8),
            )

            z_shape = linear_layer.forward(X).shape
            assert z_shape == (
                X.shape[1],
                linear_layer.u_out,
            ), f"Incorrect shape expected {(z_shape[1], linear_layer.u_out)}"

    def test_relu_correctly_applied(self):
        """
        Test case for ensuring that ReLU activation function is applied correctly.
        """

        test_inputs = [
            random_generator.normal(0, 1, size=(3, 2)),
            random_generator.normal(0, 1, size=(10, 4)),
            random_generator.normal(0, 1, size=(50, 12)),
        ]

        for X in test_inputs:

            linear_layer = LinearLayer(
                weight_init=self.random_init,
                activation_function=self.activation_fn,
                u_out=int(random_generator.uniform(1, 256) // 8),
            )

            z = linear_layer.forward(X)
            assert np.all(z >= 0), (
                "ReLU should produce only non-negative outputs. "
                "Found at least one negative value."
            )
