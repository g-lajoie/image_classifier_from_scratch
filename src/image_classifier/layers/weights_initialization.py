from abc import ABC, abstractmethod

import numpy as np
from common.enums import WeightInitiailizationMethod
from numpy import ndarray
from numpy.random import PCG64
from numpy.typing import NDArray

from image_classifier.common.variable import Variable


class WeightsInitializer(ABC):

    @abstractmethod
    def init_weights(self, X: Variable | NDArray, _out: int) -> NDArray:
        """
        X: Variable | NDArray
        _out: int
        Return: NDArray
        """
        raise NotImplementedError("The init weights method has not been created.")


class RandomInitializer(WeightsInitializer):

    def __init__(self):
        self.random = np.random.Generator(PCG64())

    def init_weights(self, X: Variable | NDArray, _out: int) -> NDArray:
        """
        Returns a random initialization of Weight Matrix(W). If X has dimensions (B,m), then W has dimensions(m,_out)

        Where:
            B = number of samples.
            m = number of features (or number of units in previous hidden layer)
            _out = number of units that will be returned from current layer.

        ---------------------
        X: Variable | NDArray
        _out: int
        Return: NDArray
        """
        # Check for correct type
        if isinstance(X, Variable):
            X = X.value

        elif isinstance(X, ndarray):
            X = X

        else:
            raise TypeError(
                f"Incorrect type of X, expected Variable or NDArray, got: {type(X)}"
            )

        p = X.shape[1]  # Number of features.

        return self.random.standard_normal(size=(p, _out))


class ScaledInitializer(WeightsInitializer):

    def __init__(self, weight_init_method: WeightInitiailizationMethod):
        """
        Initializes Scaled Initializer instance.

        --------------------------------
        weight_init_method: [Xavier, He]
            An enum of either Xaiver or He to represent the initialization method.
        """
        self.random = np.random.Generator(PCG64())

        if not isinstance(self.initializer_method, WeightInitiailizationMethod):
            raise ValueError(
                f"weights_init_method must be a member of InitializationMethod enum, got {weight_init_method}"
            )

        self.initializer_method = weight_init_method

    def init_weights(self, X: Variable | NDArray, _out: int) -> NDArray:
        """
        Returns a scaled initialization using either Xavier or He initialization methods.

        If X has dimensions (B,m), then W has dimensions(m,_out)
        Where:
            B = number of samples.
            m = number of features (or number of units in previous hidden layer)
            _out = number of units that will be returned from current layer.

        ---------------------
        X: Variable | NDArray
        _out: int
        Return: NDArray
        """

        # Check for correct type
        if isinstance(X, Variable):
            X = X.value

        elif isinstance(X, ndarray):
            X = X

        else:
            raise TypeError(
                f"Incorrect type of X, expected Variable or NDArray, got: {type(X)}"
            )

        # Select appropriate initialization method.
        if self.initializer_method == WeightInitiailizationMethod.XAVIER:
            return self.xavier_init_method(X, _out)

        elif self.initializer_method == WeightInitiailizationMethod.HE:
            return self.he_init_method(X, _out)

        else:
            raise TypeError(
                "Initialization method is missing or malformed. Please initialize object with correct initalization method"
            )

    def xavier_init_method(self, X: NDArray, _out: int) -> NDArray:
        """
        The Xavier initialization method.

        X: Input Matrix: Shape(B, m)
        Return: NDarray: Shape(m, _out)
        """

        _in = X.shape[1]  # previous layer's units, or input features.

        return self.random.normal(0, (1 / _in), size=(_in, _out))

    def he_init_method(self, X: NDArray, _out: int) -> NDArray:
        """
        The Kaising He initialization method.

        X: Input Matrix, Shape(B, m)
        Return: NDArray: Shape(m, _out)
        """

        _in = X.shape[1]  # previous layer's units, or input features.

        return self.random.normal(0, (2 / _in), size=(_in, _out))
