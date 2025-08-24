import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from numpy.random import PCG64
from numpy.typing import NDArray

from image_classifier.common.parameters import Param
from image_classifier.weight_initializers.base_weight_initialization import (
    WeightInitializationMethod,
)

logger = logging.getLogger(__name__)


class RandomInitMethod(WeightInitializationMethod):

    def __init__(self):
        self.random = np.random.Generator(PCG64())

    def init_weights(self, _in: int, _out: int) -> NDArray[np.float64]:
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

        return self.random.standard_normal(size=(_in, _out))
