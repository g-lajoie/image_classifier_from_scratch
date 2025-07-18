from typing import cast

from .base_layers import Layers
from image_classifier.functions.activiation import RELU
from image_classifier.layers import LinearLayer

class LayerStack:
    """
    Prepares sequential layers for Neural Network.

    Attributes:
        *layers: Tuple of layers.
    """

    def __init__(self, *layers: Layers):
        self.layers = layers

        # Initialization Methods
        self.set_weight_inits()

    def set_weight_inits(self):
        """
        Helper function that defines all the weight initialization methods for the layer stack.
        """

        for i in range(len(self.layers)):
            if isinstance(self.layers[i + 1], RELU) and isinstance(self.layers[i], LinearLayer):
                current_layer = cast(LinearLayer, self.layers[i])
                current_layer.weights_init =