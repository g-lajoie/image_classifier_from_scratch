import numpy as np
import pytest
from numpy.random import PCG64
from numpy.typing import NDArray

from image_classifier.common.enums.weight_initialization_enum import WeightInitMethod
from image_classifier.layers.weights_initialization import (
    RandomInitializer,
    ScaledInitializer,
)


def test_init_with_valud_enum():
    model = ScaledInitializer(WeightInitMethod.XAVIER)
    assert model.initializer_method == WeightInitMethod.XAVIER


class TestReturnType:

    random = np.random.Generator(PCG64())
    X = np.random.normal(0, 1, (12, 5))
    _out = 64

    def test_correct_return_type_random_initializer(self):
        model = RandomInitializer()
        returned_value = model.init_weights(self.X, self._out)
        assert isinstance(
            returned_value, np.ndarray
        ), f"Exepcted NDArray instead got {type(returned_value)}"

    def test_correct_return_type_scaled_initializer_xaiver(self):
        model = ScaledInitializer(weight_init_method=WeightInitMethod.XAVIER)
        returned_value = model.init_weights(self.X, self._out)
        assert isinstance(
            returned_value, np.ndarray
        ), f"Exepcted NDArray instead got {type(returned_value)}"

    def test_correct_return_type_scaled_initializer_he(self):
        model = ScaledInitializer(weight_init_method=WeightInitMethod.HE)
        returned_value = model.init_weights(self.X, self._out)
        assert isinstance(
            returned_value, np.ndarray
        ), f"Exepcted NDArray instead got {type(returned_value)}"
