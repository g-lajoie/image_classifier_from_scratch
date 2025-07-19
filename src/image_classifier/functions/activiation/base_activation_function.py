from abc import ABC, abstractmethod

from image_classifier.layers.base_layers import Layer


class ActivationFunction(Layer):
    """
    Interface for activaition class.
    """
