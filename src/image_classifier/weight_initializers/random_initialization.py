import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from numpy.random import PCG64
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.weight_initializers.base_weight_initialization import (
    BaseWeightInitializationMethod,
)

logger = logging.getLogger(__name__)


class WeightInitializationMethod(ABC):

    @abstractmethod
    def init_weights(self, X: Param, _out: int) -> NDArray:
        """
        Arguments
            X: Variable
            _out: int

        Return: NDArray
        """
        raise NotImplementedError("The init weights method has not been created.")


class RandomInitMethod(WeightInitializationMethod):

    def __init__(self):
        self.random = np.random.Generator(PCG64())

    def init_weights(self, X: Param, _out: int) -> NDArray:
        """
        Returns a random initialization of Weight Matrix(W). If X has dimensions (B,m), then W has dimensions(m,_out)

        Where:
            B = number of samples.
            m = number of features (or number of units in previous hidden layer)
            _out = number of units that will be returned from current layer.

        ---------------------
        Arguements
            X: Variable
            _out: int
        Return: NDArray
        """
        # Check for correct type
        if not isinstance(X, Param):
            raise TypeError("Invalud type for X")

        p = X.shape[1]  # Number of features.

        return self.random.standard_normal(size=(p, _out))

        return self.random.standard_normal(size=(p, _out))
