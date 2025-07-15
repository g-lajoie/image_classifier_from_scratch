import pytest

from common.enums import WeightInitiailizationMethod as InitMethod
from src.weights_initialization import ScaledInitializer


def test_init_with_valud_enum():
    model = ScaledInitializer(InitMethod.XAIVER)
    assert model.initializer_method == InitMethod.XAIVER
