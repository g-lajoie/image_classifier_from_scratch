from typing import cast

from image_classifier.layers import RELU, LinearLayer
from image_classifier.layers.base_layers import Layer
from image_classifier.weight_initializers import *


class LayerStack:
    """
    Prepares sequential layers for Neural Network.

    Attributes:
        *layers: Tuple of layers.
    """

    def __init__(self, *layers: Layer):
        self.layers = layers

        # Initialization Methods
        for idx in range(len(self.layers)):
            self.set_parent_layers(idx)
            self.set_weight_initalization_method(idx)

    def set_parent_layers(self, idx: int) -> None:
        """
        Helper function that set the parent layer.
        """

        if idx != 0:
            self.layers[idx].parent_layer = self.layers[idx - 1]

    def set_child_layers(self, idx: int) -> None:
        """
        Helper function that sets the children layer.
        """
        if idx != len(self.layers) - 1:
            self.layers[idx].child_layer = self.layers[idx + 1]

    def set_weight_initalization_method(self, idx: int) -> None:
        """
        Add the corresponding weight initialization methods for the layers
        """

        if isinstance(self.layers[idx], LinearLayer) & isinstance(
            self.layers[idx + 1], RELU
        ):
            self.layers[idx].weight_init_method = KassingInitMethod
