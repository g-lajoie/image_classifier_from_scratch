from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from numpy import ndarray
from numpy.random import PCG64

from common.enums import WeightInitiailizationMethod


class WeightsInitializer(ABC):

    @abstractmethod
    def init_weights(self, W: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
        """
        W: Weights matrix.
        b: bias vector.
        """
        raise NotImplementedError("The init weights method has not been created.")


class RandomInitializer(WeightsInitializer):

    def __init__(self):
        self.random = np.random.Generator(PCG64())

    def init_weights(self, W: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
        W = self.random.standard_normal(size=W.shape)
        b = self.random.standard_normal(size=b.shape)

        return (W, b)

class ScaledInitializer(WeightsInitializer):

    def __init__(self, weight_init_method: WeightInitiailizationMethod):
        self.random = np.random.Generator(PCG64())

        if not isinstance(self.initializer_method, WeightInitiailizationMethod):
            raise ValueError(f"weights_init_method must be a member of InitializationMethod enum, got {weight_init_method}")

        self.initializer_method =weight_init_method

    def init_weights(self, W: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
        pass

    def xaiver_init_method(self, )