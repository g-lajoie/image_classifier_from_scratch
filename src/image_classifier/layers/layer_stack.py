from typing import cast

from image_classifier.common.enums import WeightInitMethod
from image_classifier.functions.activiation import RELU
from image_classifier.layers import LinearLayer

from .base_layers import Layer


class LayerStack:
    """
    Prepares sequential layers for Neural Network.

    Attributes:
        *layers: Tuple of layers.
    """

    def __init__(self, *layers: Layer):
        self.layers = layers

        # Initialization Methods
        self.set_weight_inits()
        self.set_layer_hierarchy()

    def set_weight_inits(self):
        """
        Helper function that defines all the weight initialization methods for the layer stack.
        """

        for i in range(len(self.layers)):

            if isinstance(self.layers[i], LinearLayer):

                if i == len(self.layers) - 1:
                    pass

                if isinstance(self.layers[i + 1], RELU) and isinstance(
                    self.layers[i], LinearLayer
                ):
                    current_layer = cast(LinearLayer, self.layers[i])
                    current_layer.weight_init_method = WeightInitMethod.HE
                    current_layer.child_layer = self.layers[i + 1]

    def set_layer_hierarchy(self):
        """
        Helper function that set next layer attribute of each layer.
        """

        for i in range(len(self.layers)):
            if i != 0:
                self.layers[i].parent_layer = self.layers[i - 1]

            if i != len(self.layers) - 1:
                self.layers[i].child_layer = self.layers[i + 1]
