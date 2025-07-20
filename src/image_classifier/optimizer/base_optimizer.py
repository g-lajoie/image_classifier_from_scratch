from abc import ABC, abstractmethod

from image_classifier.layers.base_layers import Layer


class Optimizer(Layer):
    """
    Interface for optimizer classes.
    """
