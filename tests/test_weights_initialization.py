import numpy as np
import pytest
from numpy.random import PCG64
from numpy.typing import NDArray

from common.enums import WeightInitiailizationMethod as InitMethod
from src.weights_initialization import RandomInitializer, ScaledInitializer


def test_init_with_valud_enum():
    model = ScaledInitializer(InitMethod.XAIVER)
    assert model.initializer_method == InitMethod.XAIVER


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
        model = ScaledInitializer(weight_init_method=InitMethod.XAIVER)
        returned_value = model.init_weights(self.X, self._out)
        assert isinstance(
            returned_value, np.ndarray
        ), f"Exepcted NDArray instead got {type(returned_value)}"

    def test_correct_return_type_scaled_initializer_he(self):
        model = ScaledInitializer(weight_init_method=InitMethod.HE)
        returned_value = model.init_weights(self.X, self._out)
        assert isinstance(
            returned_value, np.ndarray
        ), f"Exepcted NDArray instead got {type(returned_value)}"
