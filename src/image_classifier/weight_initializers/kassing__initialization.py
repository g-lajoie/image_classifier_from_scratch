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


class KassingInitMethod(BaseWeightInitializationMethod):

    def __init__(self):
        """
        Initializes Kassing He Initializer instance.
        """

        self.random = np.random.Generator(PCG64())

    def init_weights(self, X: Param, _out: int) -> NDArray:
        """
        Returns a scaled initialization using Kassiging He initialization methods.
        To be used when the acitivation function is RELU

        If X has dimensions (B,m), then W has dimensions(m,_out)
        Where:
            B = number of samples.
            m = number of features (or number of units in previous hidden layer)
            _out = number of units that will be returned from current layer.

        ---------------------
        Arguements
            X: Variable | NDArray
            _out: int
        Return: NDArray
        """

        _in = X.shape[1]  # previous layer's units, or input features.

        return self.random.normal(0, (2 / _in), size=(_in, _out))

        return self.random.normal(0, (2 / _in), size=(_in, _out))
