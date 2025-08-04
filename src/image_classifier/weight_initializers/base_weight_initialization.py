import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from numpy.random import PCG64
from numpy.typing import NDArray

from image_classifier.common.parameters import Param

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
